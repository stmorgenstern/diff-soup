"""Warnings attributed to the user's own line.

A fixed ``stacklevel`` cannot name the caller: the number of library frames
between a warning and user code depends on the entry point.
[warn_user][diff_soup._warnings.warn_user] skips every frame inside diff_soup,
equinox, jax and jaxtyping instead, so the warning lands on the first line
outside them.
"""

from __future__ import annotations

import os
import warnings
from types import ModuleType

_PACKAGES = ("diff_soup", "equinox", "jax", "jaxtyping")
_PREFIXES: tuple[str, ...] | None = None


def _package_prefix(name: str) -> str | None:
    """Directory of the imported package ``name``, ending in a separator."""
    import importlib

    try:
        module: ModuleType = importlib.import_module(name)
    except Exception:  # pragma: no cover - package missing
        return None
    path = getattr(module, "__file__", None)
    if path is None:  # pragma: no cover - namespace package
        return None
    return os.path.join(os.path.dirname(os.path.abspath(path)), "")


def _prefixes() -> tuple[str, ...]:
    global _PREFIXES
    if _PREFIXES is None:
        found = (_package_prefix(name) for name in _PACKAGES)
        _PREFIXES = tuple(p for p in found if p is not None)
    return _PREFIXES


def warn_user(message: str, category: type[Warning] = UserWarning) -> None:
    """Emit ``message`` attributed to the first frame outside the libraries.

    Args:
        message: The warning text.
        category: The warning class.
    """
    warnings.warn(message, category, stacklevel=2, skip_file_prefixes=_prefixes())
