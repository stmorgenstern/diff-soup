"""The public setting for the sizes of the caches of compiled programs.

``ds.problem.set_cache_capacity`` sets every bounded cache of the problem layer at once, and
``ds.problem.cache_capacity`` reads them. The caches are created at import with their own
defaults, or with ``DIFFSOUP_CACHE_CAPACITY`` when it is set.
"""

from __future__ import annotations

from diff_soup.problem._cache import check_capacity, initial_capacity

#: Each cache's own default capacity, by the name `cache_capacity` gives it.
_DEFAULTS = {
    "lowering": 128,
    "value_copy": 128,
    "analysis": 128,
    "solver_view": 32,
    "solver_problem": 128,
    "recipe_builds": 128,
}


def cache_capacity() -> dict[str, int]:
    """Return the capacity of each cache of compiled programs, by name.

    Returns:
        A dictionary from the cache's name to the number of entries it keeps:
        `"lowering"`, `"value_copy"`, `"analysis"`, `"solver_view"`, `"solver_problem"`
        and `"recipe_builds"`.
    """
    from diff_soup.problem import _registry
    from diff_soup.problem._spec import recipe
    from diff_soup.problem.implicit import solver_lowering, solver_problem

    return {
        "lowering": _registry.lowering_cache.capacity,
        "value_copy": _registry.value_copy_cache.capacity,
        "analysis": _registry.analysis_cache.capacity,
        "solver_view": solver_problem._SOLVER_VIEW_CACHE.capacity,
        "solver_problem": solver_lowering._IR_SOLVER_PROBLEMS.capacity,
        "recipe_builds": recipe._BUILD_MEMO_CAPACITY,
    }


def set_cache_capacity(n: int | None) -> None:
    """Set the number of entries every cache of compiled programs keeps.

    The caches hold lowered problems, solver views and the inputs a recipe has built, and a
    program compiled for an entry is kept with it. A larger `n` keeps more of them across
    changes of structure. A smaller `n` evicts the least recently used entries down to it.
    The defaults are 128, and 32 for solver views. `DIFFSOUP_CACHE_CAPACITY` sets `n` for a
    process from the start.

    Args:
        n: The number of entries, at least 1.

    Raises:
        TypeError: If `n` is not an `int`.
        ValueError: If `n` is less than 1.
    """
    if n is not None:
        check_capacity(n, "n")
    from diff_soup.problem import _registry
    from diff_soup.problem._spec import recipe
    from diff_soup.problem.implicit import solver_lowering, solver_problem

    size = {name: initial_capacity(default) if n is None else n for name, default in _DEFAULTS.items()}
    _registry.lowering_cache.capacity = size["lowering"]
    _registry.value_copy_cache.capacity = size["value_copy"]
    _registry.analysis_cache.capacity = size["analysis"]
    solver_problem._SOLVER_VIEW_CACHE.capacity = size["solver_view"]
    solver_problem._SOLVER_VIEW_CACHE_CAPACITY = size["solver_view"]
    solver_lowering._IR_SOLVER_PROBLEMS.capacity = size["solver_problem"]
    recipe._BUILD_MEMO_CAPACITY = size["recipe_builds"]
