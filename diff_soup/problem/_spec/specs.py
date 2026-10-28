# Declarative spec records produced by the ProblemSpec builder.
"""Declarative spec records produced by the ProblemSpec builder.

Plain frozen dataclasses (not ``eqx.Module``) — these are host-side
configuration consumed by
[build][diff_soup.problem._spec.problem_spec.ProblemSpec.build] to emit a frozen
``EquationSet`` IR.  They never cross JIT trace boundaries.  Each record is
constructed once and then read; field mutation is not supported.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from diff_soup.domains.boundary_set import BoundarySet


@dataclass(frozen=True)
class FieldSpec:
    """A user-declared spatial field (lowers to a ``Variable``).

    ``scale`` / ``offset`` populate the IR
    [Scales][diff_soup.problem._ir.scales.Scales] (``state_SI = offset + scale ·
    state_internal``); ``None`` means "omitted" (defaults to ``1.0`` and triggers
    the un-scaled warning) and is distinct from an explicit ``scale=1.0``
    (intentional dimensionless). ``residual_scale`` divides this field's whole
    equation row — the assembled residual and, for a differential field, the
    mass-diagonal slice — so the row's Jacobian entries shrink while its
    solution stays the same.

    ``init`` is an optional default spatial initial condition: a callable
    taking one positional coordinate array per spatial dim (``*xs``), evaluated
    over the field's domain node coordinates ``(num_nodes, dim)``.  The IC stays
    *decoupled* — it is still an argument to
    [assemble_state][diff_soup.problem._spec.problem_spec.ProblemSpec.assemble_state]
    — but when no explicit value is supplied for this field, ``init`` populates
    its ``y0`` block.  ``init`` wins over the scalar ``default`` and is overridden
    by an explicit ``values[name]``.
    """

    name: str
    mass: Literal["differential", "algebraic"]
    shape: Optional[tuple[int, ...]] = None
    bounds: Optional[tuple[Optional[float], Optional[float]]] = None
    scale: Optional[float] = None
    offset: Optional[float] = None
    residual_scale: Optional[float] = None
    init: Optional[Callable[..., Any]] = None
    # FEM strong-form authoring opt-in.  When ``True`` and the field's domain
    # is FEM, ``q.<name>`` returns a quad-native
    # [Field][diff_soup.domains.operators._field.Field] (decays to the
    # quadrature-sampled array under arithmetic / ``jnp`` ufuncs, exposes
    # ``.nodal`` for operators).  Inert on grids and for non-opted fields
    # (``q.<name>`` returns a plain array — the default behaviour).
    quad: bool = False

    def __repr__(self) -> str:
        return f"FieldSpec({self.name!r}, mass={self.mass!r})"


@dataclass(frozen=True)
class GlobalVarSpec:
    """A user-declared scalar or vector global (lowers to a ``GlobalVariable``).

    ``scale`` populates the IR [Scales][diff_soup.problem._ir.scales.Scales]
    state reference under the global sentinel namespace ``("", name)``.  ``None``
    means omitted (defaults to ``1.0``; globals never trigger the un-scaled
    warning).

    ``mass`` records the global's structural role.  An ``"algebraic"``
    global (the default) lowers to a multiplier in the ``lam`` slot whose
    row is the constraint or control row that closes it, and that row
    carries no mass diagonal.  A ``"differential"`` global is written by
    exactly one ordinary equation returning its time derivative and
    carries unit mass; validation refuses one declared alongside a
    closing constraint or a bound control.
    """

    name: str
    initial: Any = 0.0
    mass: Literal["algebraic", "differential"] = "algebraic"
    scale: Optional[float] = None

    def __repr__(self) -> str:
        return f"GlobalVarSpec({self.name!r}, mass={self.mass!r})"


@dataclass(frozen=True, eq=False)
class ParamSpec:
    """A user-declared parameter and its stable public handle.

    Equality and hashing use only ``(namespace, kind, name)``. In particular,
    they never compare ``value`` arrays, so the record can be passed unchanged
    through parameter selection, fitting, and UQ result lookup.
    """

    name: str
    value: Any
    learnable: bool = False
    bounds: Optional[tuple[Optional[float], Optional[float]]] = None
    prior: Optional[Any] = None
    namespace: str = dataclass_field(default="", metadata={"fingerprint": False})

    @property
    def identity(self) -> tuple[str, str, str]:
        """Stable in-process identity shared by every inference front end."""
        return (self.namespace, "parameter", self.name)

    @property
    def kind(self) -> str:
        """Return ``"scalar"`` or ``"array"`` for the physical value."""
        return "scalar" if np.shape(self.value) == () else "array"

    @property
    def shape(self) -> tuple[int, ...]:
        """Physical array shape; a scalar has shape ``()``."""
        return tuple(np.shape(self.value))

    def __hash__(self) -> int:
        """Hash by semantic declaration identity, never by array value."""
        return hash(self.identity)

    def __eq__(self, other: object) -> bool:
        """Compare declaration identity without invoking array equality."""
        return isinstance(other, ParamSpec) and self.identity == other.identity

    def __repr__(self) -> str:
        return f"ParamSpec({self.name!r}, learnable={self.learnable!r})"


@dataclass(frozen=True, eq=False)
class FunctionSpec:
    """A named function declaration and its stable public handle."""

    name: str
    fn: Callable
    learnable: bool = False
    prior: Optional[Any] = None
    namespace: str = dataclass_field(default="", metadata={"fingerprint": False})
    #: ``fn`` maps one sample, so ``q.fn.<name>`` applies it at every node.
    pointwise: bool = False

    @property
    def identity(self) -> tuple[str, str, str]:
        """Stable in-process identity shared by every inference front end."""
        return (self.namespace, "function", self.name)

    @property
    def kind(self) -> str:
        """Return the public parameter kind."""
        return "function"

    @property
    def domain(self) -> Optional[tuple[float, float]]:
        """Physical input interval declared by the function approximator."""
        interval = getattr(self.fn, "interval", None)
        if interval is None:
            return None
        return (float(interval[0]), float(interval[1]))

    def __hash__(self) -> int:
        """Hash by semantic declaration identity, never callable identity."""
        return hash(self.identity)

    def __eq__(self, other: object) -> bool:
        """Compare declaration identity without comparing callable leaves."""
        return isinstance(other, FunctionSpec) and self.identity == other.identity

    def __repr__(self) -> str:
        return f"FunctionSpec({self.name!r}, learnable={self.learnable!r})"


@dataclass(frozen=True)
class EquationSpec:
    """A user-declared residual equation (lowers to an ``Equation``)."""

    writes: str
    reads: tuple[str, ...]
    fn: Callable
    name: str


@dataclass(frozen=True)
class MultiEquationSpec:
    """A multi-write residual closure (lowers to a ``MultiEquation``).

    The closure ``fn(q)`` returns a dict ``{var_name: residual}`` covering
    every output in ``writes``.  Use when several equations share an
    expensive per-cell kernel (FEM element residuals, AC-mech, ...) so
    the kernel runs exactly once per residual call rather than once per
    output variable.
    """

    writes: tuple[str, ...]
    reads: tuple[str, ...]
    fn: Callable
    name: str


@dataclass(frozen=True)
class BCSpec:
    """A user-declared boundary condition (lowers to a discriminated IR BC node)."""

    kind: Literal[
        "dirichlet",
        "flux",
        "robin",
        "traction",
        "symmetry",
        "periodic",
    ]
    var: str
    where: dict[str, Any]
    value: Optional[Any] = None
    value_fn: Optional[Callable] = None
    # Dirichlet only: which value of a multi-component node the pin replaces.
    # ``None`` pins every value the node carries.
    component: Optional[int] = None
    # Field-dependent boundary flux: ``fn(face, t)`` reading a boundary-restricted
    # ``face`` view.  ``name`` is an optional stable handle a flux-total closure
    # links to.  Mutually exclusive with ``value`` / ``value_fn``.
    fn: Optional[Callable] = None
    name: Optional[str] = None
    # Field-dependent flux locality: ``False`` (the default) evaluates ``fn``
    # once per facet, so a cross-facet read is impossible by construction;
    # ``True`` hands ``fn`` the whole boundary and declares the conservative
    # boundary-wide Jacobian pattern.
    cross_facet: bool = False
    # Robin fields
    alpha: Optional[Any] = None
    beta: Optional[Any] = None


@dataclass(frozen=True)
class ConstraintSpec:
    """A user-declared global constraint (lowers to ``Constraint``)."""

    kind: Literal[
        "integral",
        "galvanostatic",
        "reduce_residual",
        "boundary_integral",
        "flux_total",
    ]
    name: str
    closes: Optional[str] = None
    field: Optional[str] = None
    integrand_fn: Optional[Callable] = None
    target_fn: Optional[Callable] = None
    current_fn: Optional[Callable] = None
    reduce: Optional[Callable] = None
    tag: Optional[str] = None
    selector: Optional[BoundarySet] = None
    integrand_scale: Optional[float] = None
    # ``flux_total`` link: the boundary-flux BC names whose integrals are summed.
    from_bc: Optional[tuple[str, ...]] = None


@dataclass(frozen=True)
class CouplingSpec:
    """A user-declared cross-component data transfer.

    Populated by [couple][diff_soup.problem._spec.problem_spec.ProblemSpec.couple],
    whose ``as_=`` form records a read-through coupling. Consumed by the build walk
    and the lowering provider.

    ``read_through`` (default ``False``) marks a NO-DOF read-through coupling
    (``couple(..., as_=)``: the source value is injected by name into the consuming
    component's closure with no allocated state DOF).  ``target_owner`` (default
    ``None``) names the component that consumes the alias; it is carried onto the IR
    [Coupling][diff_soup.problem._ir.coupling.Coupling] so an alias read only by a
    boundary-condition callback (and by no equation) still resolves its owner.
    """

    source: Any
    target: Any
    via: Any  # TransferOp instance
    name: str
    read_through: bool = False
    target_owner: Optional[str] = None


@dataclass(frozen=True)
class ObservationSpec:
    """A user-declared observation (lowers to ``Observation``).

    ``domain`` names the component whose grid and operator bank the observation
    evaluates with; ``None`` lets the build pick the component whose fields the
    observation reads.
    """

    name: str
    kind: Literal["scalar", "field", "boundary_mean", "spatial_mean"]
    fn: Callable
    where: Optional[dict[str, Any]] = None
    domain: Optional[str] = None


@dataclass(frozen=True)
class ControlSpec:
    """A user-declared time-dependent control (lowers to ``Control``).

    ``kind="direct"`` prescribes the value outright and carries no
    ``binds_to``; ``kind="derived"`` drives an assembled quantity and names
    the global that carries it.
    """

    name: str
    kind: Literal["direct", "derived"]
    fn: Callable
    binds_to: Optional[str] = None


@dataclass(frozen=True)
class OperationalControlSpec:
    """A user-declared operational control table.

    ``current`` / ``voltage`` name declared scalar observations; ``closes``
    names the algebraic global the table's runtime-selected residual row
    closes.  ``modes`` is the static ordered mode-name tuple the runtime
    mode index selects from.  ``seed_voltage_threshold`` bounds the terminal
    voltage magnitude below which the power-mode default current seed
    ``target / V`` is not taken.  ``field`` names the field whose assembled
    residual rows sum to the current when the table takes its current from
    them (``current`` is then ``None``); it is ``None`` when ``current``
    names an observation.
    """

    name: str
    current: Optional[str]
    voltage: str
    closes: str
    modes: tuple[str, ...]
    seed_voltage_threshold: float = 1e-6
    field: Optional[str] = None
