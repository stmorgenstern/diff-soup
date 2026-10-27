"""Loader for the KLU host sparse LU extension.

The native module ``_native_klu`` is built per checkout from
``native/klu_ffi.cpp`` against the SuiteSparse KLU shared libraries (see
``CMakeLists.txt``). [load][diff_soup._internal.klu_ffi.load] imports it once
and registers its three XLA FFI targets on the CPU platform:

- ``diffsoup_klu_factor_f64``: factor values into a slot;
- ``diffsoup_klu_refactor_f64``: refactor into a slot where a predicate holds;
- ``diffsoup_klu_apply_f64``: solve with a slot's factors, in either orientation.

[available][diff_soup._internal.klu_ffi.available] reports whether the module
imports, without raising. The module's other functions (the analysis, site and
context registry, the counters) are reached as attributes of this package.
"""

from __future__ import annotations

import importlib
import threading
from types import ModuleType
from typing import Optional

_BUILD_HINT = (
    "The KLU extension diff_soup._internal.klu_ffi._native_klu is not built in this "
    "checkout. Build it from the repository root against the SuiteSparse tree in "
    "diff_soup/_internal/klu_ffi/vendor (or pass -DKLU_VENDOR_DIR=<tree>) with:\n"
    "    cmake -S diff_soup/_internal/klu_ffi -B build/klu_ffi\n"
    "    cmake --build build/klu_ffi -j\n"
)

_LOCK = threading.Lock()
_NATIVE: Optional[ModuleType] = None


def load() -> ModuleType:
    """Import the native module and register its FFI targets (once per process).

    Returns:
        The ``_native_klu`` module.

    Raises:
        ImportError: If the module is not built in this checkout or its
            SuiteSparse libraries cannot be loaded, with the commands that
            build it.
    """
    global _NATIVE
    with _LOCK:
        if _NATIVE is not None:
            return _NATIVE
        try:
            native = importlib.import_module("diff_soup._internal.klu_ffi._native_klu")
        except ImportError as exc:
            raise ImportError(f"{_BUILD_HINT}Import failed with: {exc}") from exc
        import jax

        for name, capsule in native.registrations().items():
            jax.ffi.register_ffi_target(name, capsule, platform="cpu")
        _NATIVE = native
        return native


def available() -> bool:
    """Whether the native module imports in this checkout.

    Returns:
        ``True`` when [load][diff_soup._internal.klu_ffi.load] succeeds.
    """
    try:
        load()
    except ImportError:
        return False
    return True


def __getattr__(name: str):
    """Resolve a native attribute (a registry function or a counter) from the loaded module."""
    if name.startswith("_"):
        raise AttributeError(name)
    native = load()
    try:
        return getattr(native, name)
    except AttributeError:
        raise AttributeError(
            f"module 'diff_soup._internal.klu_ffi' has no attribute {name!r}"
        ) from None
