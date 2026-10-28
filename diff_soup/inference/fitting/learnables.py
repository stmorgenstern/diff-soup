"""Declared-learnable discovery and named selection.

A model states which of its quantities are unknown. A Recipe states it with
``spec.param(name, value=..., learnable=True)`` and
``spec.function(name, fn, learnable=True)``; a hand-written model states it by
exposing a ``learnable_paths()`` method naming its own fitted leaves. Nothing
else is a declaration: an array leaf is not a decision variable because it
happens to be an array.

Everything here reads those declarations. A fit that states no selection of its
own fits exactly the declared learnables, a name matches a declared learnable,
and a leaf that was never declared is reached only by naming it explicitly in a
[ParameterGroups][diff_soup.inference.fitting.inference_problem.ParameterGroups].
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any, Optional

import equinox as eqx
import jax

from diff_soup.inference.fitting._path_filter import _walk_path, parameters_at
from diff_soup.problem._ir.content_hash import declared_callable_layers, iter_declared_closures


def _dotted_key_path(path: tuple[Any, ...]) -> str:
    """Convert a JAX pytree key path to the dotted selector grammar."""
    parts: list[str] = []
    for key in path:
        if hasattr(key, "name"):
            part = key.name
        elif hasattr(key, "idx"):
            part = key.idx
        elif hasattr(key, "key"):
            part = key.key
        else:  # pragma: no cover - protects custom pytree key implementations
            raise TypeError(f"unsupported pytree path key {type(key).__name__}")
        parts.append(str(part))
    return ".".join(parts)


def _module_leaf_field_names(obj: Any) -> tuple[str, ...]:
    """Return every nested inexact-array path on an ``eqx.Module``."""
    if not isinstance(obj, eqx.Module) or not dataclasses.is_dataclass(obj):
        return ()
    filtered = eqx.filter(obj, eqx.is_inexact_array)
    return tuple(
        _dotted_key_path(path)
        for path, leaf in jax.tree_util.tree_leaves_with_path(filtered)
        if leaf is not None
    )


def _function_leaf_paths(fn: Any) -> tuple[str, ...]:
    """Return the leaf paths a declared callable contributes, relative to itself.

    A callable that declares ``learnable_paths()`` is read at its word — an
    approximator that holds both fitted coefficients and fixed data names only
    the fitted ones. Otherwise every nested inexact-array leaf of the
    ``eqx.Module`` is taken, including weights inside lists of layers.
    """
    if hasattr(fn, "learnable_paths"):
        paths = tuple(str(p) for p in fn.learnable_paths())
        if paths:
            return paths
    return _module_leaf_field_names(fn)


def _traced_adapter_layer(problem: Any) -> Optional[tuple[str, Any]]:
    """Return ``(dotted_path, adapter)`` for one lowered closure adapter.

    Every lowered closure of one Recipe shares a single params namespace and a
    single functions namespace, so one adapter answers for the whole problem.
    Returns ``None`` when the problem carries no such adapter (a model authored
    without the Recipe layer, or one that declares nothing learnable).
    """
    for path, fn in iter_declared_closures(problem):
        if fn is None:
            continue
        for suffix, layer in declared_callable_layers(fn):
            if type(layer).__name__ == "_TracedQAdapter":
                return (f"{path}.{suffix}" if suffix else path), layer
    return None


def _model_declared_paths(problem: Any) -> tuple[str, ...]:
    """Return the leaf paths a model declares for itself, or ``()``."""
    declare = getattr(problem, "learnable_paths", None)
    if not callable(declare):
        return ()
    return tuple(str(path) for path in declare())


def learnable_paths(problem: Any, names: Iterable[str] | None = None) -> list[str]:
    """Return the internal leaf paths of ``problem``'s declared learnables.

    Public for diagnostics; the paths themselves stay implementation details of
    the lowering. Each path resolves to one array leaf of ``problem`` and is
    named by its declaration: a Recipe parameter or function by the name it was
    declared under, a model-level declaration by the path the model states.

    Args:
        problem: A lowered problem, or a model declaring ``learnable_paths()``.
        names: The declared names to return paths for, or ``None`` for every
            declared learnable.

    Returns:
        The leaf paths, in declaration order.

    Raises:
        ValueError: If a requested name was never declared learnable, or a
            declared learnable function exposes no array leaves.
    """
    requested = None if names is None else {str(n) for n in names}
    paths: list[str] = []
    found: set[str] = set()
    declared: list[str] = []

    for path in _model_declared_paths(problem):
        name = path.rsplit(".", 1)[-1]
        declared.append(name)
        if requested is None or name in requested:
            paths.append(path)
            found.add(name)

    located = _traced_adapter_layer(problem)
    if located is not None:
        base, adapter = located
        declared.extend(getattr(adapter, "learnable_params", ()))
        declared.extend(getattr(adapter, "learnable_functions", ()))
        for name in getattr(adapter, "learnable_params", ()):
            if requested is None or name in requested:
                paths.append(f"{base}.params.{name}")
                found.add(name)
        functions = getattr(adapter, "functions", None)
        for name in getattr(adapter, "learnable_functions", ()):
            if requested is not None and name not in requested:
                continue
            fn = functions[name] if functions is not None and name in functions else None
            leaf_paths = _function_leaf_paths(fn)
            if not leaf_paths:
                raise ValueError(
                    f"learnable function {name!r} exposes no array leaves to fit. "
                    "Declare it with an eqx.Module holding array fields, or give "
                    "the module a learnable_paths() naming its fitted leaves."
                )
            paths.extend(f"{base}.functions.{name}.{leaf}" for leaf in leaf_paths)
            found.add(name)

    if requested is not None:
        missing = sorted(requested - found)
        if missing:
            raise ValueError(
                f"name(s) {missing} are not declared learnable on this problem. "
                f"Declared learnables: {sorted(set(declared))}. Declare a parameter "
                "with spec.param(..., learnable=True) or a function with "
                "spec.function(..., learnable=True)."
            )

    return paths


def declared_learnable_leaf_ids(problem: Any) -> frozenset:
    """Return the ``id()`` set of the array leaves ``problem`` declares learnable.

    One declared value can be referenced by several lowered closures — one
    parameter read by three equations is one array object sitting at three
    places in the pytree — so identity, not position, is what selects it: every
    occurrence of the declared value answers to the same declaration.
    """
    return frozenset(
        id(leaf)
        for path in learnable_paths(problem, None)
        for leaf in jax.tree_util.tree_leaves(_walk_path(problem, path))
        if eqx.is_inexact_array(leaf)
    )


def learnable_function_leaf_ids(problem: Any) -> frozenset:
    """Return the ``id()`` set of the array leaves of ``problem``'s learnable functions.

    These are the weights of the networks and other callables declared with
    ``spec.function(..., learnable=True)``, a subset of
    [declared_learnable_leaf_ids][diff_soup.inference.fitting.learnables.declared_learnable_leaf_ids].
    """
    located = _traced_adapter_layer(problem)
    if located is None or not getattr(located[1], "learnable_functions", ()):
        return frozenset()
    prefix = f"{located[0]}.functions."
    return frozenset(
        id(leaf)
        for path in learnable_paths(problem, None)
        if path.startswith(prefix)
        for leaf in jax.tree_util.tree_leaves(_walk_path(problem, path))
        if eqx.is_inexact_array(leaf)
    )


@dataclasses.dataclass(frozen=True)
class LearnableSelector:
    """A named selection of a problem's declared learnables.

    Returned by [learnables][diff_soup.inference.fitting.learnables.learnables].
    Carries the requested parameter names (used
    to label a parameter group and to build the optimizer's decision-variable
    selection) and the problem-structured boolean mask (used to partition the
    problem pytree).  Pass it as a
    [ParameterGroups][diff_soup.inference.fitting.inference_problem.ParameterGroups]
    group value.

    Attributes:
        names: The requested names (``None`` selects every declared learnable).
        mask: A boolean pytree matching the problem structure, ``True`` at the
            selected leaves.
    """

    names: Optional[tuple]
    mask: Any

    def __iter__(self):
        """Yield the requested names (empty when ``names`` is ``None``)."""
        return iter(() if self.names is None else self.names)


def learnables(problem: Any, names: Iterable[str] | None = None) -> Optional[LearnableSelector]:
    """Select the learnable parameters and functions declared on a built problem.

    Args:
        problem: The built problem.
        names: The declared names to select, or `None` to select every learnable.

    Returns:
        A selector whose `mask` is a boolean pytree over `problem` marking the selected arrays, or `None` when nothing is learnable.
    """
    paths = learnable_paths(problem, names)
    if not paths:
        return None
    requested = None if names is None else tuple(str(n) for n in names)
    return LearnableSelector(names=requested, mask=parameters_at(problem, paths))


def has_declared_learnables(problem: Any) -> bool:
    """Return ``True`` if ``problem`` declares any learnable parameter or function."""
    return bool(learnable_paths(problem, None))


__all__ = [
    "learnables",
    "learnable_paths",
    "declared_learnable_leaf_ids",
    "has_declared_learnables",
    "LearnableSelector",
]
