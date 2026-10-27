"""Value-independent structural identity of a solver problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Optional

import numpy as np

__all__ = ["StructureSignature"]


def _topology_signature(topology) -> Optional[tuple]:
    if topology is None:
        return None
    rows = np.asarray(topology.rows, dtype=np.int32)
    cols = np.asarray(topology.cols, dtype=np.int32)
    return (
        int(topology.n_dofs),
        tuple(int(size) for size in rows.shape),
        rows.tobytes(),
        tuple(int(size) for size in cols.shape),
        cols.tobytes(),
    )


def declared_structure_key(source: Any) -> Optional[Hashable]:
    """The structure key ``source`` declares, or ``None`` when it declares none.

    Raises:
        TypeError: If the declared key is not hashable.
    """
    if source is None:
        return None
    getter = getattr(source, "structure_key", None)
    candidate = getter() if callable(getter) else getattr(source, "_structure_key", None)
    if candidate is None:
        return None
    try:
        hash(candidate)
    except TypeError as exc:
        raise TypeError(
            "structure_key must return hashable, value-independent static data."
        ) from exc
    return candidate


def _declared_source_key(source: Any, problem: Any) -> Hashable:
    """Use an explicit structural key or conservatively fall back to identity."""
    candidate = declared_structure_key(source)
    if candidate is not None:
        return ("declared", candidate)
    owner = problem if source is None else source
    return ("identity", id(owner))


def _control_signature(control_bindings: Any) -> tuple:
    if control_bindings is None:
        return ()
    return tuple(
        (
            "derived",
            str(binding.control_name),
            str(binding.global_name),
            int(binding.lam_offset),
            int(binding.dim),
        )
        for binding in tuple(control_bindings.bindings)
    ) + tuple(
        ("direct", str(entry.control_name))
        for entry in tuple(getattr(control_bindings, "direct", ()))
    )


@dataclass(frozen=True)
class StructureSignature:
    """Compile-relevant solver-problem structure, excluding runtime numeric values.

    ``source_key`` names the model the structure belongs to: ``("declared",
    structure_key)`` for a source that declares a value-independent structure
    key, ``("content", digest)`` for a model known by a fingerprint of its
    content (its values included, since they are compiled into its program),
    and ``("identity", id)`` for a model known only as an object.
    """

    source_key: Hashable
    problem_kind: str
    plan_mode: str
    n: int
    n_u: int
    n_lam: int
    id_mask: bytes
    partitions: tuple
    fy_topology: Optional[tuple]
    fydot_topology: Optional[tuple]
    full_topology: Optional[tuple]
    theta_structure: tuple
    control_bindings: tuple

    @classmethod
    def build(
        cls,
        *,
        source: Any,
        problem: Any,
        plan: Any,
        n: int,
        n_u: int,
        n_lam: int,
        id_mask: Any,
        partitions: tuple,
        theta_structure: tuple,
        control_bindings: Any,
        source_key: Optional[Hashable] = None,
    ) -> "StructureSignature":
        """Capture one hashable signature from concrete host-side structure.

        ``source_key`` defaults to the source's declared structure key, or
        the identity of the source (the problem when there is none).
        """
        problem_cls = type(problem)
        return cls(
            source_key=_declared_source_key(source, problem) if source_key is None else source_key,
            problem_kind=f"{problem_cls.__module__}.{problem_cls.__qualname__}",
            plan_mode=str(plan.mode),
            n=int(n),
            n_u=int(n_u),
            n_lam=int(n_lam),
            id_mask=np.asarray(id_mask, dtype=np.int32).tobytes(),
            partitions=tuple(partitions),
            fy_topology=_topology_signature(plan.fy_topology),
            fydot_topology=_topology_signature(plan.fydot_topology),
            full_topology=_topology_signature(plan.union_topology),
            theta_structure=theta_structure,
            control_bindings=_control_signature(control_bindings),
        )
