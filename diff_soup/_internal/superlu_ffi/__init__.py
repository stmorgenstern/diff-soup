"""Loader for the SuperLU_MT host sparse LU extension.

The native module ``_native_superlu`` is built per checkout from
``native/superlu_solver.cpp`` against a SuperLU_MT installation (see
``CMakeLists.txt``). [load][diff_soup._internal.superlu_ffi.load] imports it
once and registers its XLA FFI targets on the CPU platform, one of each for
``float64`` (``_f64``) and for ``complex128`` (``_c128``) values:

- ``diffsoup_superlu_factor_*``: factor values into a slot;
- ``diffsoup_superlu_refactor_*``: refactor into a slot where a predicate holds;
- ``diffsoup_superlu_apply_*``: solve with a slot's factors, in either orientation.

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

    The factor and refactor targets keep a state per call of a compiled program, whose type is
    registered with XLA first.

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

        # XLA refuses a handler whose state type it does not know, and it applies the pending
        # registrations of a platform handlers first when it makes the platform's client, so
        # the CPU client is made before the state type and the handlers are registered.
        try:
            jax.devices("cpu")
        except RuntimeError:  # the CPU platform is disabled; nothing can run the handlers
            pass
        state_type = getattr(native, "call_state_type", None)
        if state_type is not None:  # a module built before the calls had their state has none
            jax.ffi.register_ffi_type("diffsoup_superlu_call_state", state_type(), platform="cpu")
        for name, handler in native.registrations().items():
            jax.ffi.register_ffi_target(name, handler, platform="cpu")
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
