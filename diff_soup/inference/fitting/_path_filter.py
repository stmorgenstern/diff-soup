"""Path-based pytree-leaf selector for ``learnable_filter`` arguments.

The default learnable filter captures leaf identity via `id` at
filter-construction time.  That breaks under `equinox.tree_at`,
which produces a fresh pytree with new leaf objects.

`parameters_at` instead returns a **mask pytree** of the same
structure as the input, with ``True`` at the targeted leaves and
``False`` elsewhere.  The mask survives `equinox.tree_at` because
the rewrite preserves tree structure -- only the leaf values change.

Use with `equinox.partition` / `equinox.combine` exactly as
you would a callable filter::

    mask = parameters_at(problem, [
        "components.0.equations.0.params.D",
        "globals.0.init",
    ])
    learnable, frozen = eqx.partition(problem, mask)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import jax


def _walk_path(root: Any, path: str) -> Any:
    """Walk a dotted path through ``root`` using getattr / __getitem__.

    Numeric path components (e.g. ``"0"``) trigger ``__getitem__``; all
    other components use `getattr`.  Raises `AttributeError`
    or `IndexError` / `KeyError` on mismatch.
    """
    obj = root
    for part in path.split("."):
        try:
            idx = int(part)
        except ValueError:
            try:
                obj = getattr(obj, part)
            except AttributeError:
                if isinstance(obj, Mapping) and part in obj:
                    obj = obj[part]
                else:
                    raise
        else:
            obj = obj[idx]
    return obj


def parameters_at(pytree: Any, paths: Iterable[str]) -> Any:
    r"""Build a learnable-filter mask selecting leaves at given dotted paths.

    Each entry in ``paths`` is a dotted attribute path; integer-looking
    components index into tuples / lists / sequences via
    `__getitem__`.  All other components use `getattr`.  The
    resolved value at each path must be a JAX-leaf in ``pytree`` (i.e.
    appear in `jax.tree.leaves`); paths to interior pytree nodes
    raise `ValueError`.

    Args:
        pytree: The pytree to walk.  Typically a `LoweredProblem`
            or an `EquationSet`.
        paths: Iterable of dotted paths.  Each path identifies one leaf.

    Returns:
        A pytree of bools matching ``pytree``'s tree structure.  A leaf
        is ``True`` exactly when one of ``paths`` resolves to that leaf.

    Raises:
        ValueError: A path resolves to an interior pytree node, or
            does not resolve to any leaf in ``pytree``.

    Example::

        from diff_soup import parameters_at
        import equinox as eqx

        # Pick out a single learnable scalar.
        mask = parameters_at(problem, ["components.0.equations.0.params.D"])
        learnable, frozen = eqx.partition(problem, mask)

        # The mask survives tree_at because only leaf values change:
        new_problem = eqx.tree_at(
            lambda p: p.components[0].equations[0].params.D,
            problem,
            replace=jnp.array(0.5),
        )
        learnable_2, _ = eqx.partition(new_problem, mask)  # still works
    """
    paths = list(paths)
    if not paths:
        # Empty selector -> all-False mask
        return jax.tree.map(lambda _: False, pytree)

    target_ids = set()
    for path in paths:
        leaf = _walk_path(pytree, path)
        target_ids.add(id(leaf))

    leaves = jax.tree.leaves(pytree)
    leaf_ids = {id(leaf) for leaf in leaves}
    missing = target_ids - leaf_ids
    if missing:
        # Build a friendlier message: report the offending paths.
        bad_paths = []
        for path in paths:
            try:
                leaf = _walk_path(pytree, path)
            except (AttributeError, IndexError, KeyError, TypeError):
                bad_paths.append(path)
                continue
            if id(leaf) not in leaf_ids:
                bad_paths.append(path)
        raise ValueError(
            f"parameters_at: paths do not resolve to JAX leaves of pytree: "
            f"{bad_paths}. Each path must reach an array-like leaf "
            f"(static fields and non-array attrs are not selectable)."
        )

    return jax.tree.map(lambda leaf: id(leaf) in target_ids, pytree)


__all__ = ["parameters_at"]
