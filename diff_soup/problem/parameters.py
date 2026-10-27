"""Handle-based parameter selection shared by optimization and UQ."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


def _validate_handles(values: Any, *, where: str, declaration: str = "ParameterSet") -> tuple:
    """Validate and freeze one sequence of Recipe parameter handles.

    ``declaration`` names the declaring class in the refusals.
    """
    try:
        handles = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{declaration} {where} must be an iterable of Recipe handles.") from exc
    for handle in handles:
        if not hasattr(handle, "identity") or not hasattr(handle, "name"):
            raise TypeError(
                f"{declaration} {where} accepts only handles returned by "
                "spec.param(...) or spec.function(...); got "
                f"{type(handle).__name__}."
            )
        if not bool(getattr(handle, "learnable", False)):
            raise ValueError(
                f"{declaration} cannot fit non-learnable handle {handle.name!r}; "
                "declare it with learnable=True."
            )
    return handles


@dataclasses.dataclass(frozen=True, init=False)
class ParameterSet:
    """The learnable parameters of a fit, split into those shared by every experiment and those varying by cluster.

    Pass it as `parameters=` of
    [`InferenceProblem`][diff_soup.inference.InferenceProblem]. Every parameter
    must be learnable, come from one recipe, and appear once.

    Args:
        shared: Parameters returned by `spec.param` or `spec.function` that take one value across every experiment.
        varying: Mapping from cluster name to the parameters that take one value per cluster.
    """

    shared: tuple
    varying: Mapping[str, tuple]

    def __init__(self, *, shared: Any = (), varying: Mapping[str, Any] | None = None):
        """Validate ownership and freeze the declaration."""
        shared_handles = _validate_handles(shared, where="shared")
        if varying is None:
            varying = {}
        if not isinstance(varying, Mapping):
            raise TypeError("ParameterSet varying must map cluster names to Recipe handles.")

        varying_handles: dict[str, tuple] = {}
        for cluster, values in varying.items():
            if not isinstance(cluster, str) or not cluster:
                raise TypeError(
                    f"ParameterSet varying cluster names must be non-empty strings; got {cluster!r}."
                )
            varying_handles[cluster] = _validate_handles(values, where=f"varying[{cluster!r}]")

        all_handles = shared_handles + tuple(
            handle for values in varying_handles.values() for handle in values
        )
        if not all_handles:
            raise ValueError("ParameterSet requires at least one handle.")
        identities = [handle.identity for handle in all_handles]
        duplicates = sorted(
            {handle.name for handle in all_handles if identities.count(handle.identity) > 1}
        )
        if duplicates:
            raise ValueError(
                f"ParameterSet handle(s) {duplicates} appear more than once; every "
                "parameter must have exactly one ownership declaration."
            )

        namespaces = {handle.identity[0] for handle in all_handles}
        if len(namespaces) > 1:
            raise ValueError(
                "ParameterSet handles must come from one Recipe declaration; "
                "mixing same-named handles from independent specs is ambiguous."
            )
        names = [handle.name for handle in all_handles]
        repeated_names = sorted({name for name in names if names.count(name) > 1})
        if repeated_names:
            raise ValueError(
                f"ParameterSet handle names must be unique; got duplicate name(s) "
                f"{repeated_names}. Parameter names label optimizer groups and "
                "cannot be ambiguous across scalar and function declarations."
            )

        object.__setattr__(self, "shared", shared_handles)
        object.__setattr__(self, "varying", MappingProxyType(varying_handles))

    @property
    def handles(self) -> tuple:
        """Every handle in deterministic flat-schema order."""
        return self.shared + tuple(handle for values in self.varying.values() for handle in values)

    def cluster_for(self, handle: Any) -> str | None:
        """Return the varying cluster for ``handle``, or ``None`` if shared."""
        if any(candidate == handle for candidate in self.shared):
            return None
        for cluster, handles in self.varying.items():
            if any(candidate == handle for candidate in handles):
                return cluster
        raise KeyError(f"ParameterSet does not contain handle {getattr(handle, 'name', handle)!r}.")


@dataclasses.dataclass(frozen=True)
class ParameterOccurrence:
    """One shared or owner-specific occurrence in a flat decision vector."""

    owner: Any
    columns: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class ParameterEntry:
    """Schema entry for one Recipe parameter handle."""

    handle: Any
    occurrences: tuple[ParameterOccurrence, ...]
    size: int
    decode: Callable[[Any], Any] = dataclasses.field(compare=False, repr=False)


def _key_path(path: tuple[Any, ...]) -> str:
    """Render a JAX pytree key path in the learnable-path grammar."""
    parts = []
    for key in path:
        if hasattr(key, "name"):
            parts.append(str(key.name))
        elif hasattr(key, "idx"):
            parts.append(str(key.idx))
        elif hasattr(key, "key"):
            parts.append(str(key.key))
        else:  # pragma: no cover - protects third-party pytree key types
            raise TypeError(f"unsupported pytree path key {type(key).__name__}")
    return ".".join(parts)


def _function_parameter_partition(fn: Any) -> tuple[Any, Any]:
    """Partition one function module at exactly its declared learnable paths."""
    declare = getattr(fn, "learnable_paths", None)
    paths = tuple(str(path) for path in declare()) if callable(declare) else ()
    if not paths:
        return eqx.partition(fn, eqx.is_inexact_array)

    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(fn)
    mask = jax.tree_util.tree_unflatten(
        treedef,
        [
            eqx.is_inexact_array(leaf)
            and any(name == path or name.startswith(f"{path}.") for path in paths)
            for key_path, leaf in path_leaves
            for name in (_key_path(key_path),)
        ],
    )
    return eqx.partition(fn, mask)


def parameter_decoder(handle: Any) -> tuple[int, Callable[[Any], Any]]:
    """Return the scalar size and physical-value decoder for a Recipe handle."""
    if getattr(handle, "kind", None) in ("scalar", "array"):
        shape = tuple(getattr(handle, "shape", np.shape(handle.value)))
        size = int(np.prod(shape, dtype=int)) if shape else 1

        def decode(values):
            value = jnp.reshape(jnp.asarray(values), shape)
            return value

        return size, decode

    if getattr(handle, "kind", None) == "function":
        dynamic, static = _function_parameter_partition(handle.fn)
        flat, unravel = ravel_pytree(dynamic)
        size = int(flat.size)
        if size == 0:
            raise ValueError(
                f"learnable function handle {handle.name!r} exposes no inexact array leaves."
            )

        def decode(values):
            return eqx.combine(unravel(jnp.asarray(values)), static)

        return size, decode

    raise TypeError(
        f"unsupported parameter handle kind {getattr(handle, 'kind', None)!r} "
        f"for {getattr(handle, 'name', handle)!r}."
    )


@dataclasses.dataclass(frozen=True, init=False)
class ParameterSchema:
    """Mapping between Recipe parameter handles and a fitted flat coordinate."""

    entries: tuple[ParameterEntry, ...]
    fingerprint: str

    def __init__(self, entries: Any, *, fingerprint: str):
        """Validate non-overlapping entries and freeze their order."""
        entries = tuple(entries)
        identities = [entry.handle.identity for entry in entries]
        if len(set(identities)) != len(identities):
            raise ValueError("ParameterSchema entries must have unique handle identities.")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "fingerprint", str(fingerprint))

    @property
    def handles(self) -> tuple:
        """The parameter handles in flat-schema order."""
        return tuple(entry.handle for entry in self.entries)

    def entry(self, handle: Any) -> ParameterEntry:
        """Resolve by identity, falling back to the declared name and decoded size."""
        identity = getattr(handle, "identity", None)
        for entry in self.entries:
            if entry.handle.identity == identity:
                return entry
        name = getattr(handle, "name", None)
        candidates = [entry for entry in self.entries if entry.handle.name == name]
        if candidates:
            size, _ = parameter_decoder(handle)
            matching = [entry for entry in candidates if entry.size == size]
            if len(matching) == 1:
                return matching[0]
        raise KeyError(
            f"parameter handle {getattr(handle, 'name', handle)!r} does not belong "
            "to this fit's schema."
        )

    def value(self, handle: Any, flat: Any) -> Any:
        """Reconstruct one shared value or an owner-keyed varying mapping."""
        entry = self.entry(handle)
        vector = jnp.asarray(flat)
        values = tuple(
            (occurrence.owner, entry.decode(vector[jnp.asarray(occurrence.columns)]))
            for occurrence in entry.occurrences
        )
        if len(values) == 1 and values[0][0] is None:
            return values[0][1]
        return {owner: value for owner, value in values}


__all__ = [
    "ParameterSet",
    "ParameterSchema",
    "ParameterEntry",
    "ParameterOccurrence",
    "parameter_decoder",
]
