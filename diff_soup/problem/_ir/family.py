"""Family IR node — a population of structurally-identical members.

A [Family][diff_soup.problem._ir.family.Family] describes a population of structurally-identical members
(e.g. a particle ensemble for an MPM electrode): one shared per-member
``domain`` and per-member variable / equation / BC / coupling templates,
replicated ``count`` times.  The per-member parameter arrays (``member_params``
— e.g. ``D_s`` / ``R``) are pytree leaves so they remain differentiable and
vmappable.

Initial conditions are decoupled from the IR: they are supplied at
``assemble_state`` time, so there is deliberately no ``member_init`` field.

``domain`` is a runtime (non-static) pytree field — it carries JAX arrays
(nodes, faces, ...) that cannot live behind ``eqx.field(static=True)``.  This
mirrors [Component][diff_soup.problem._ir.component.Component], whose ``domain`` is also a leaf.  The family
geometry is still consumed structurally at lowering/host time (never
differentiated); its identity is folded by ``content_hash._family_id`` via
``_domain_id``.
"""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import numpy as np

from .component import validate_operator_factory
from .family_graph import FamilyGraph, FamilyGraphExchange


def _validate_stencil_radius(name, stencil_radius):
    """Return ``stencil_radius`` as ``None`` or a positive ``int``.

    Raises:
        TypeError: for a non-integer (including ``bool``) declaration.
        ValueError: for a non-positive radius.
    """
    if stencil_radius is None:
        return None
    if isinstance(stencil_radius, bool) or not isinstance(stencil_radius, (int, np.integer)):
        raise TypeError(
            f"family {name!r}: stencil_radius must be None or a positive int "
            f"(cells of Jacobian reach per axis); received "
            f"{type(stencil_radius).__name__}."
        )
    radius = int(stencil_radius)
    if radius < 1:
        raise ValueError(
            f"family {name!r}: stencil_radius must be at least 1 (received "
            f"{radius}); omit it to declare the member domain's own stencil."
        )
    return radius


class Family(eqx.Module):
    """Record of a population of `count` identical members on one member domain.

    `spec.family(...)` builds one. Names within a family must be unique, and
    every graph must have the family's member count.

    Args:
        name: Family name, non-empty.
        count: Number of members.
        domain: Member domain shared by every member.
        member_variables: Member field variables.
        member_equations: Member equations.
        member_bcs: Member flux boundary conditions.
        member_couplings: Couplings owned by the family, left empty by the Recipe API.
        scale: State scale of each member field, keyed by field name.
        residual_scale: Residual divisor of each member field, keyed by field name.
        operator_factory: Operator-bank class or top-level named function of the domain, or `None` for the built-in bank.
        member_params: Per-member parameter arrays, keyed by name.
        graphs: [`FamilyGraph`][diff_soup.problem.FamilyGraph] records over the members.
        graph_exchanges: [`FamilyGraphExchange`][diff_soup.problem.FamilyGraphExchange] terms along those graphs.
        stencil_radius: Jacobian reach of a member kernel in cells per axis, or `None` for one cell.
    """

    name: str = eqx.field(static=True)
    count: int = eqx.field(static=True)
    # ``domain`` is a runtime (pytree) leaf — it carries JAX arrays so it cannot
    # live behind ``static=True``.  Mirrors Component; the geometry is consumed
    # structurally at lowering time and its identity is folded by
    # ``content_hash._family_id`` via ``_domain_id``.
    domain: object
    member_variables: tuple = eqx.field(static=True, default=())
    member_equations: tuple = eqx.field(static=True, default=())
    member_bcs: tuple = eqx.field(static=True, default=())
    member_couplings: tuple = eqx.field(static=True, default=())
    scale: object = eqx.field(static=True, default=None)
    residual_scale: object = eqx.field(static=True, default=None)
    # Operator-bank factory selecting the spatial operators each member is
    # lowered against (an operator-bank class or a top-level named callable
    # ``domain -> bank``).  Stored opaquely as static structural configuration;
    # ``None`` selects the built-in bank.  The lowering resolver constructs and
    # type-checks the bank from the family's current member ``domain``.
    operator_factory: object = eqx.field(static=True, default=None)
    # Global index of this population's FIRST member.  ``0`` for a standalone
    # family; a ragged family lowered as several static size buckets gives each
    # bucket the running offset so ``m.member_index`` stays the member's
    # position in the WHOLE authored population.
    member_index_offset: int = eqx.field(static=True, default=0)
    # Declared per-member stencil reach (``None`` = the member domain's own
    # radius-1 node-pair pattern; ``r >= 1`` = every node within ``r`` cells
    # per axis).  Static structural configuration consumed by the sparsity
    # builder; folded into the content hash and the serialized manifest.
    stencil_radius: object = eqx.field(static=True, default=None)
    member_params: Mapping[str, "jax.Array"] = eqx.field(default_factory=dict)
    # runtime (non-static) tuple of flux callables for the member
    # FluxBCs.  Parallel to ``member_params`` — kept as pytree leaves so AD /
    # vmap reach any learnable array fields inside the callable (e.g. MLP weights).
    # Each entry is the ``flux_fn`` from the corresponding FluxBC in
    # ``member_bcs`` (matching index); lowering refuses a member BC of any other
    # kind, so the two stay index-aligned.
    # Default: empty tuple → the no-callable path.
    member_flux_callables: tuple = ()
    # Canonical adjacency records and exchange callables. Both are non-static
    # pytrees: graph topology/declarations live in each node's static treedef,
    # while numeric weights and learnable callable leaves remain differentiable.
    graphs: tuple = ()
    graph_exchanges: tuple = ()

    def __init__(
        self,
        *,
        name: str,
        count: int,
        domain,
        member_variables: tuple = (),
        member_equations: tuple = (),
        member_bcs: tuple = (),
        member_couplings: tuple = (),
        scale=None,
        residual_scale=None,
        operator_factory: object = None,
        member_index_offset: int = 0,
        member_params: Mapping[str, "jax.Array"] | None = None,
        member_flux_callables: tuple | None = None,
        graphs: tuple = (),
        graph_exchanges: tuple = (),
        stencil_radius: int | None = None,
    ):
        validate_operator_factory(operator_factory)
        self.stencil_radius = _validate_stencil_radius(name, stencil_radius)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("family name must be a non-empty string.")
        self.name = name
        self.count = int(count)
        self.member_index_offset = int(member_index_offset)
        # Runtime (pytree) field — its JAX arrays remain in the leaf partition.
        self.domain = domain
        self.member_variables = tuple(member_variables)
        self.member_equations = tuple(member_equations)
        self.member_couplings = tuple(member_couplings)
        self.scale = scale
        self.residual_scale = residual_scale
        self.operator_factory = operator_factory
        self.member_params = dict(member_params) if member_params is not None else {}
        # Keep flux callables in the runtime tuple so learnable eqx.Module array
        # leaves remain visible to AD. The BC tuple retains a stripped structural
        # copy for content identity without putting arrays in static fields.
        from diff_soup.problem._ir.bc import FluxBC as _FluxBC  # noqa: PLC0415

        if member_flux_callables is not None:
            self.member_flux_callables = tuple(member_flux_callables)
        else:
            self.member_flux_callables = tuple(
                bc.flux_fn for bc in member_bcs if isinstance(bc, _FluxBC)
            )
        normalized_graphs = tuple(graphs)
        graph_names: set[str] = set()
        for graph in normalized_graphs:
            if not isinstance(graph, FamilyGraph):
                raise TypeError(
                    f"family {self.name!r}: graphs entries must be FamilyGraph "
                    f"objects; received {type(graph).__name__}. Construct each "
                    "entry with FamilyGraph(...) or family.graph(...)."
                )
            if graph.name in graph_names:
                raise ValueError(
                    f"family {self.name!r}: duplicate graph name {graph.name!r}; "
                    "give every graph a unique name."
                )
            graph_names.add(graph.name)
            graph_member_count = graph.member_count
            if graph_member_count != self.count:
                raise ValueError(
                    f"family {self.name!r}: graph {graph.name!r} "
                    f"has member_count {graph_member_count!r}, but the family count is "
                    f"{self.count}; construct the graph for this family population."
                )
        normalized_exchanges = tuple(graph_exchanges)
        exchange_names: set[str] = set()
        field_names = {getattr(variable, "name", None) for variable in self.member_variables}
        for exchange in normalized_exchanges:
            if not isinstance(exchange, FamilyGraphExchange):
                raise TypeError(
                    f"family {self.name!r}: graph_exchanges entries must be "
                    f"FamilyGraphExchange objects; received {type(exchange).__name__}. "
                    "Construct each entry with FamilyGraphExchange(...) or "
                    "family.graph_exchange(...)."
                )
            if exchange.name in exchange_names:
                raise ValueError(
                    f"family {self.name!r}: duplicate graph exchange name "
                    f"{exchange.name!r}; give every exchange a unique name."
                )
            exchange_names.add(exchange.name)
            if exchange.graph not in graph_names:
                raise ValueError(
                    f"family {self.name!r}: graph exchange {exchange.name!r} names "
                    f"unknown graph {exchange.graph!r}; include that graph in "
                    "graphs= before declaring the exchange."
                )
            for label, dependencies in (
                ("writes", (exchange.writes,)),
                ("reads", exchange.reads),
                ("weight_reads", exchange.weight_reads),
            ):
                unknown = next((name for name in dependencies if name not in field_names), None)
                if unknown is not None:
                    raise ValueError(
                        f"family {self.name!r}: graph exchange {exchange.name!r} "
                        f"{label} names unknown member field {unknown!r}; declare "
                        "the field in member_variables or remove the dependency."
                    )
        self.graphs = normalized_graphs
        self.graph_exchanges = normalized_exchanges

        # Store member_bcs in the static slot with JAX arrays stripped from any
        # eqx.Module flux_fn callables. content_hash._eqx_module_fingerprint
        # already ignores numeric leaves (uses eqx.partition(mod, eqx.is_array) to get
        # the static partition), so the stripped version produces an identical fingerprint.
        # Stripping avoids the equinox "A JAX array is being set as static!" warning
        # without call-site suppressions or module-level filters.
        def _strip_bc(bc):
            if not isinstance(bc, _FluxBC):
                return bc
            flux_fn = bc.flux_fn
            if isinstance(flux_fn, eqx.Module):
                # Replace array leaves with None — same type+structure for fingerprinting.
                stripped_fn = eqx.filter(flux_fn, lambda x: not eqx.is_array(x))
                return _FluxBC(
                    var=bc.var,
                    where=bc.where,
                    flux_fn=stripped_fn,
                    bc_name=bc.bc_name,
                    cross_facet=bool(getattr(bc, "cross_facet", False)),
                )
            return bc

        self.member_bcs = tuple(_strip_bc(bc) for bc in member_bcs)

    @property
    def member_state_size(self) -> int:
        """State size of a single member.

        Sums ``size`` over ``member_variables`` when those templates carry a
        ``size`` attribute (the
        [Variable][diff_soup.problem._ir.variable.Variable] convention).
        Falls back to
        ``0`` for an empty / size-less template tuple.
        """
        total = 0
        for v in self.member_variables:
            size = getattr(v, "size", None)
            if size is not None:
                total += int(size)
        return int(total)

    @property
    def state_size(self) -> int:
        """Total state size of the population: ``count * member_state_size``."""
        return int(self.count) * self.member_state_size
