"""Loader for the SuperLU_MT host sparse LU extension.

The native module ``_native_superlu`` is built per checkout from
``native/superlu_solver.cpp`` against a SuperLU_MT installation (see
``CMakeLists.txt``). [load][diff_soup._internal.superlu_ffi.load] imports it
once and registers its three XLA FFI targets on the CPU platform:

- ``diffsoup_superlu_factor_f64``: factor values into a slot;
- ``diffsoup_superlu_refactor_f64``: refactor into a slot where a predicate holds;
- ``diffsoup_superlu_apply_f64``: solve with a slot's factors, in either orientation.

[available][diff_soup._internal.superlu_ffi.available] reports whether the
module imports, without raising.
"""

from __future__ import annotations

import importlib
import threading
from types import ModuleType
from typing import Optional

_BUILD_HINT = (
    "The SuperLU_MT extension diff_soup._internal.superlu_ffi._native_superlu is not built in "
    "this checkout. Install SuperLU_MT (OpenMP build, -fPIC, its internal BLAS) into a prefix, "
    "then build the module from the repository root with:\n"
    "    cmake -S diff_soup/_internal/superlu_ffi -B build/superlu_ffi "
    "-DSUPERLU_MT_ROOT=<prefix>\n"
    "    cmake --build build/superlu_ffi -j\n"
)

_LOCK = threading.Lock()
_NATIVE: Optional[ModuleType] = None
_IMPORT_ERROR: Optional[BaseException] = None


def load() -> ModuleType:
    """Import the native module and register its FFI targets (once per process).

    Returns:
        The ``_native_superlu`` module.

    Raises:
        ImportError: If the module is not built in this checkout, with the
            commands that build it.
    """
    global _NATIVE, _IMPORT_ERROR
    with _LOCK:
        if _NATIVE is not None:
            return _NATIVE
        try:
            native = importlib.import_module("diff_soup._internal.superlu_ffi._native_superlu")
        except ImportError as exc:
            _IMPORT_ERROR = exc
            raise ImportError(_BUILD_HINT) from exc
        import jax

        for name, capsule in native.registrations().items():
            jax.ffi.register_ffi_target(name, capsule, platform="cpu")
        _NATIVE = native
        return native


def available() -> bool:
    """Whether the native module imports in this checkout.

    Returns:
        ``True`` when [load][diff_soup._internal.superlu_ffi.load] succeeds.
    """
    try:
        load()
    except ImportError:
        return False
    return True
