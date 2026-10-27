"""Static family adjacency and conservative exchange declarations.

The edge index is structural: it is validated and canonicalized on the host,
then stored as immutable integer tuples. Numeric edge weights remain ordinary
pytree leaves so changing their values does not change content identity or the
compiled topology. A callable weight rule is retained on the pytree in the
same way as equation and boundary-condition callables.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np


def _canonical_edges(
    edges: Any, *, member_count: int, directed: bool, graph_name: str
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Validate and canonicalize an authored ``(2, E)`` edge index."""
    try:
        raw = np.asarray(edges)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"family graph {graph_name!r}: edges must be an integer array with shape (2, E)."
        ) from exc
    if raw.ndim != 2 or raw.shape[0] != 2:
        raise ValueError(
            f"family graph {graph_name!r}: edges has shape {raw.shape}, but it "
            "must have shape (2, E) with source indices in row 0 and target "
            "indices in row 1."
        )
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.bool_):
        raise TypeError(
            f"family graph {graph_name!r}: edges must use an integer dtype; received {raw.dtype}."
        )
    values = raw.astype(np.int64, copy=True)
    if values.size:
        bad = values[(values < 0) | (values >= int(member_count))]
        if bad.size:
            raise ValueError(
                f"family graph {graph_name!r}: member index {int(bad[0])} is out "
                f"of range; every edge endpoint must be in [0, {int(member_count)})."
            )
        loops = np.flatnonzero(values[0] == values[1])
        if loops.size:
            node = int(values[0, int(loops[0])])
            raise ValueError(
                f"family graph {graph_name!r}: self-loop ({node}, {node}) is not "
                "supported; remove self-loops and express member-local terms in "
                "a member equation."
            )
        if not directed:
            lo = np.minimum(values[0], values[1])
            hi = np.maximum(values[0], values[1])
            values = np.stack((lo, hi))
        order = np.lexsort((values[1], values[0]))
        values = values[:, order]
        if values.shape[1] > 1:
            duplicate = np.flatnonzero(
                (values[0, 1:] == values[0, :-1]) & (values[1, 1:] == values[1, :-1])
            )
            if duplicate.size:
                pos = int(duplicate[0])
                edge = (int(values[0, pos]), int(values[1, pos]))
                qualifier = "reversed or repeated" if not directed else "repeated"
                raise ValueError(
                    f"family graph {graph_name!r}: duplicate edge {edge} is "
                    f"{qualifier}; declare every canonical edge exactly once."
                )
    else:
        order = np.empty((0,), dtype=np.int64)
    return (
        tuple(int(x) for x in values[0]),
        tuple(int(x) for x in values[1]),
        tuple(int(x) for x in order),
    )


def _normalize_weights(
    weights: Any,
    *,
    edge_count: int,
    canonical_order: tuple[int, ...],
    graph_name: str,
) -> Any:
    """Keep a callable rule or normalize numeric scalar/edge-array weights."""
    if callable(weights):
        return weights
    try:
        value = jnp.asarray(weights)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"family graph {graph_name!r}: weights must be a numeric scalar, a "
            f"numeric array with length {edge_count}, or a callable weight rule."
        ) from exc
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError(
            f"family graph {graph_name!r}: numeric weights must use a real "
            f"floating-point dtype; received {value.dtype}. Use floating-point "
            "weights (for example float32 or float64) so they remain in the "
            "differentiable parameter pytree."
        )
    if value.ndim == 0:
        return value
    if value.ndim != 1 or int(value.shape[0]) != int(edge_count):
        raise ValueError(
            f"family graph {graph_name!r}: weight array length must equal the "
            f"edge count {edge_count}; received shape {tuple(value.shape)}."
        )
    return value[jnp.asarray(canonical_order, dtype=jnp.int32)]


class FamilyGraph(eqx.Module):
    """Edge list connecting members of one family, with a weight per edge.

    `family.graph(...)` builds one. Undirected edges are stored with the lower
    member index first, and self-loops and duplicate edges raise `ValueError`.

    Args:
        name: Graph name, unique within the family.
        member_count: Number of members, the family's `count`.
        edges: Integer array of shape `(2, E)`, source member indices in row 0 and targets in row 1.
        directed: Whether each edge has a direction, given explicitly.
        weights: A float scalar, a float array of length `E`, or a function `(source, target, edge)` returning each edge's weight.
    """

    name: str = eqx.field(static=True)
    member_count: int = eqx.field(static=True)
    directed: bool = eqx.field(static=True)
    edge_sources: tuple[int, ...] = eqx.field(static=True)
    edge_targets: tuple[int, ...] = eqx.field(static=True)
    weights: object

    def __init__(
        self,
        *,
        name: str,
        member_count: int,
        edges: Any,
        directed: bool,
        weights: Any,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("family graph name must be a non-empty string.")
        if not isinstance(directed, (bool, np.bool_)):
            raise TypeError(f"family graph {name!r}: directed must be an explicit bool.")
        if not isinstance(member_count, (int, np.integer)) or int(member_count) <= 0:
            raise ValueError(f"family graph {name!r}: member_count must be a positive integer.")
        sources, targets, canonical_order = _canonical_edges(
            edges,
            member_count=int(member_count),
            directed=bool(directed),
            graph_name=name,
        )
        self.name = name
        self.member_count = int(member_count)
        self.directed = bool(directed)
        self.edge_sources = sources
        self.edge_targets = targets
        self.weights = _normalize_weights(
            weights,
            edge_count=len(sources),
            canonical_order=canonical_order,
            graph_name=name,
        )

    @property
    def edges(self) -> np.ndarray:
        """Canonical integer edge index with shape ``(2, E)``."""
        value = np.asarray((self.edge_sources, self.edge_targets), dtype=np.int32)
        value.flags.writeable = False
        return value

    @property
    def edge_count(self) -> int:
        """Number of canonical edges."""
        return len(self.edge_sources)


class FamilyGraphExchange(eqx.Module):
    """Pairwise exchange term along the edges of a family graph.

    `family.graph_exchange(...)` builds one.

    Args:
        name: Exchange name, unique within the family.
        graph: Name of the [`FamilyGraph`][diff_soup.problem.FamilyGraph] the exchange runs along.
        writes: Member field whose rows receive the exchange.
        reads: Member fields `fn` reads at both endpoints.
        weight_reads: Member fields a weight function reads.
        fn: Callable `(source, target, edge)` returning the flow into the target.
    """

    name: str = eqx.field(static=True)
    graph: str = eqx.field(static=True)
    writes: str = eqx.field(static=True)
    reads: tuple[str, ...] = eqx.field(static=True)
    weight_reads: tuple[str, ...] = eqx.field(static=True)
    fn: Callable

    def __init__(
        self,
        *,
        name: str,
        graph: str,
        writes: str,
        reads: tuple[str, ...],
        weight_reads: tuple[str, ...],
        fn: Any,
    ) -> None:
        def nonempty(value: Any, *, label: str) -> str:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"family graph exchange {label} must be a non-empty string.")
            return value

        def dependencies(value: Any, *, label: str) -> tuple[str, ...]:
            if isinstance(value, str):
                raise TypeError(
                    f"family graph exchange {label} must be a sequence of field "
                    f"names, not the string {value!r}."
                )
            try:
                normalized = tuple(value)
            except TypeError as exc:
                raise TypeError(
                    f"family graph exchange {label} must be a sequence of field names."
                ) from exc
            if any(not isinstance(item, str) or not item.strip() for item in normalized):
                raise ValueError(
                    f"family graph exchange {label} entries must be non-empty field names."
                )
            if len(set(normalized)) != len(normalized):
                raise ValueError(
                    f"family graph exchange {label} contains duplicate dependencies; "
                    "declare each field once."
                )
            return normalized

        exchange_name = nonempty(name, label="name")
        if not callable(fn):
            raise TypeError(
                f"family graph exchange {exchange_name!r} fn must be callable; "
                "pass a function or callable equinox.Module accepting "
                "(source, target, edge)."
            )
        self.name = exchange_name
        self.graph = nonempty(graph, label="graph")
        self.writes = nonempty(writes, label="writes")
        self.reads = dependencies(reads, label="reads")
        self.weight_reads = dependencies(weight_reads, label="weight_reads")
        self.fn = fn


__all__ = ["FamilyGraph", "FamilyGraphExchange"]
