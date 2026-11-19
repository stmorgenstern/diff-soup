"""Loader for the member chain extension, the threaded block Thomas kernel on the host.

The native module ``_native_member_chain`` is built with the KLU extension from ``native/member_chain.cpp`` (see
``CMakeLists.txt``). [load][diff_soup._internal.member_chain_ffi.load] imports it once and registers its XLA FFI
targets on the CPU platform, one of each for ``float64`` (``_f64``) and ``complex128`` (``_c128``) values:

- ``diffsoup_member_chain_factor_*``: factor a batch of block-tridiagonal chains;
- ``diffsoup_member_chain_solve_*``: solve with that factor, in either orientation.

The targets are stateless. [available][diff_soup._internal.member_chain_ffi.available] reports whether the module
imports, without raising.
"""

from __future__ import annotations

import importlib
import threading
from types import ModuleType
from typing import Optional

_BUILD_HINT = (
    "The member chain extension diff_soup._internal.member_chain_ffi._native_member_chain is not built in "
    "this checkout. It is built with the KLU extension; from the repository root:\n"
    "    cmake -S diff_soup/_internal -B build/native -DDIFF_SOUP_BUILD_CUDSS=OFF "
    "-DDIFF_SOUP_BUILD_SUPERLU=OFF\n"
    "    cmake --build build/native -j\n"
)

_LOCK = threading.Lock()
_NATIVE: Optional[ModuleType] = None


def load() -> ModuleType:
    """Import the native module and register its FFI targets (once per process).

    Returns:
        The ``_native_member_chain`` module.

    Raises:
        ImportError: If the module is not built in this checkout, with the commands that build it.
    """
    global _NATIVE
    with _LOCK:
        if _NATIVE is not None:
            return _NATIVE
        try:
            native = importlib.import_module("diff_soup._internal.member_chain_ffi._native_member_chain")
        except ImportError as exc:
            raise ImportError(_BUILD_HINT) from exc
        import jax

        for name, handler in native.registrations().items():
            jax.ffi.register_ffi_target(name, handler, platform="cpu")
        _NATIVE = native
        return native


def available() -> bool:
    """Whether the native module imports in this checkout.

    Returns:
        ``True`` when [load][diff_soup._internal.member_chain_ffi.load] succeeds.
    """
    try:
        load()
    except ImportError:
        return False
    return True
