"""The closures of a control horizon, kept per model so a new problem reuses a compiled solve."""

from __future__ import annotations

import collections
from collections.abc import Callable
from typing import Any

#: How many (model, configuration) entries are kept; the oldest is dropped first.
_CAPACITY = 32

#: ``(id(model), key) -> (model, closures)``; holding the model keeps its id from being reused.
_ENTRIES: collections.OrderedDict = collections.OrderedDict()


def cached(model: Any, key: tuple, build: Callable[[], Any]) -> Any:
    """The closures `build()` gives for `model` and `key`, built once while they stay cached."""
    entry_key = (id(model), key)
    entry = _ENTRIES.get(entry_key)
    if entry is not None and entry[0] is model:
        _ENTRIES.move_to_end(entry_key)
        return entry[1]
    closures = build()
    _ENTRIES[entry_key] = (model, closures)
    while len(_ENTRIES) > _CAPACITY:
        _ENTRIES.popitem(last=False)
    return closures
