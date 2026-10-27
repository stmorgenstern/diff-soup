"""Bounded content-keyed cache for the problem layer.

Every memo of the problem layer is an instance of
[BoundedContentCache][diff_soup.problem._cache.BoundedContentCache]: the
lowering, value-copy and analysis caches of [diff_soup.problem._registry][],
the solver-view cache, the solver-problem cache of lowered Recipe IR and the
coloring caches of the sparse Jacobian. Each instance lives as a module global
in the module that builds its values.

Design invariants
-----------------
* **Key = content (lookup) + handle (diagnostic).** A hit returns the
  *same* built value object, so a caller keyed on ``id(value)`` (a solver view,
  a ``filter_jit`` compile) reuses what it built for it.  ``lower()`` keys its
  structure analysis by the content hash and the lowered problems it hands out
  by the content hash and a digest of their declared values
  ([LoweredProblem][diff_soup.problem._ir.LoweredProblem]).
* **The key discriminates deliberate recompiles.** The hash covers IR
  structure, static metadata, ``Scales``, and ``Layout`` / ``Precision``
  (folded in by the caller via
  [compose_content_key][diff_soup.problem._cache.compose_content_key]).
  A changed key is a cache miss (a rebuild), never a silent reuse.
* **Bounded.** LRU with an explicit capacity and eviction telemetry.
* **Thread safe.** One lock per instance serializes lookup, build and insert,
  so concurrent callers of one key observe one built object.

Why two key parts?  ``content_hash`` is the primary lookup key: two
structurally identical specs share a built value.  The ``handle`` is stored
alongside the value and never takes part in the lookup; a caller whose key
falls back to an object's identity passes the object itself as the handle, so
the entry keeps it alive and its ``id`` cannot be reused while the entry lives.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from threading import RLock
from typing import Any, Optional


class BoundedContentCache:
    """Bounded LRU cache keyed by ``content_hash`` with id-stable values.

    Every cache of the problem layer is an instance of this class, held as a
    module global next to the code that builds its values.

    A cache hit returns the *same value object* that was stored on the
    corresponding miss, so callers that key ``filter_jit`` on ``id(value)``
    reuse their compiled function across calls (the id-stability guarantee).
    Eviction follows strict LRU on access order; the eviction count is exposed
    for telemetry.

    Attributes:
        capacity: Maximum number of retained entries before LRU eviction.
    """

    def __init__(self, capacity: int = 128) -> None:
        """Initialise an empty bounded cache.

        Args:
            capacity: Maximum retained entries before LRU eviction. Must be
                a positive integer.

        Raises:
            ValueError: If ``capacity`` is not a positive integer.
        """
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(
                f"BoundedContentCache capacity must be a positive int, got {capacity!r}"
            )
        self._capacity: int = capacity
        # Maps content_hash -> (handle, value). OrderedDict gives O(1) LRU:
        # move_to_end on access, popitem(last=False) on eviction.
        self._store: OrderedDict[Hashable, tuple[Any, Any]] = OrderedDict()
        self._evictions: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._lock = RLock()

    @property
    def capacity(self) -> int:
        """Maximum number of retained entries before LRU eviction."""
        return self._capacity

    def get_or_build(
        self,
        content_hash: Hashable,
        handle: Any,
        build_fn: Callable[[], Any],
    ) -> Any:
        """Return the cached value for the key, or build + insert it.

        A cache hit returns the id-stable value object for an unchanged
        ``content_hash`` — the *same* object is returned, avoiding an XLA
        recompile cascade.  A miss (a ``content_hash`` not present, including
        one whose ``Layout`` / ``Precision`` was folded in differently) calls
        ``build_fn``, inserts the result, and evicts the least-recently-used
        entry when at capacity.

        Args:
            content_hash: Structural / metadata / ``Scales`` / ``Layout`` /
                ``Precision`` content hash (the deliberate-recompile
                discriminant). Must be hashable.
            handle: Stored alongside the value and never part of the lookup
                (``content_hash`` is the lookup key). Pass the keyed object
                itself when the key holds its ``id``, so the entry keeps it
                alive.
            build_fn: Zero-arg builder invoked on a miss. Its return value is
                cached and returned id-stably on subsequent hits.

        Returns:
            The cached or freshly-built value (the SAME object across hits).
        """
        with self._lock:
            entry = self._store.get(content_hash)
            if entry is not None:
                self._hits += 1
                # LRU touch: most-recently-used moves to the end.
                self._store.move_to_end(content_hash)
                return entry[1]
            self._misses += 1
            value = build_fn()
            self._store[content_hash] = (handle, value)
            self._store.move_to_end(content_hash)
            self._evict_to_capacity()
            return value

    def _evict_to_capacity(self) -> None:
        """Evict least-recently-used entries until size <= capacity."""
        while len(self._store) > self._capacity:
            self._store.popitem(last=False)  # pop the LRU (front) entry
            self._evictions += 1

    def __contains__(self, content_hash: Hashable) -> bool:
        """Return whether ``content_hash`` is cached (no LRU touch)."""
        return content_hash in self._store

    def __len__(self) -> int:
        """Return the number of retained entries."""
        return len(self._store)

    def evictions(self) -> int:
        """Return the cumulative eviction count (SolverStats-adjacent telemetry).

        Returns:
            The number of entries evicted by LRU since construction.
        """
        return self._evictions

    def hits(self) -> int:
        """Return the cumulative cache-hit count (telemetry)."""
        return self._hits

    def misses(self) -> int:
        """Return the cumulative cache-miss count (telemetry)."""
        return self._misses

    def clear(self) -> None:
        """Drop all entries and reset telemetry counters.

        Use this as a test or session-reset hook.  The registry exposes
        [reset_registry][diff_soup.problem._registry.reset_registry] which calls this on
        every owned cache so tests start from a clean state.  Counters reset
        alongside the store so per-test telemetry is isolated.
        """
        with self._lock:
            self._store.clear()
            self._evictions = 0
            self._hits = 0
            self._misses = 0


def compose_content_key(
    structure_hash: Hashable,
    *,
    layout: Optional[Any] = None,
    precision: Optional[Any] = None,
    scales: Optional[Any] = None,
) -> Hashable:
    """Fold ``Layout`` / ``Precision`` / ``Scales`` into a structural hash.

    The content-hash cache key covers IR structure (``structure_hash``,
    computed over the ``EquationSet``) together with the layout, precision, and
    scaling policies.  Changing any of these produces a different composed key,
    triggering a deliberate recompile rather than a silent reuse.

    ``Layout`` and ``Precision`` are hashable by construction; ``Scales`` is a
    frozen dataclass of plain scalars.  Default-constructed ``Layout()`` and
    ``Precision()`` hash identically across independent constructions, so a
    default-policy build re-hits its cache entry rather than recompiling.

    Args:
        structure_hash: The hashable structural fingerprint of the IR.
        layout: Optional [Layout][diff_soup._layout.Layout] to fold in.
        precision: Optional [Precision][diff_soup._precision.Precision] to fold in.
        scales: Optional [Scales][diff_soup.problem._ir.Scales] to fold in.

    Returns:
        A composite hashable key suitable for
        [BoundedContentCache.get_or_build][diff_soup.problem._cache.BoundedContentCache.get_or_build].
    """
    return (structure_hash, layout, precision, scales)
