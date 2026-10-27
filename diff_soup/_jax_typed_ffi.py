"""Write typed-FFI custom-call configuration onto the StableHLO op itself.

Every foreign-function (FFI) kernel a JAX program calls -- JAX's own LAPACK
and cuSOLVER routines (``lapack_dgetrf_ffi``, ``cusolver_getrf_ffi``,
``cu_lu_pivots_to_permutation``, ...) as much as diff_soup's cuDSS kernels
(``diffsoup_reuse_*``, ``diffsoup_solve_*``) -- lowers to a
``stablehlo.custom_call`` whose configuration is a dictionary of attributes
and whose calling convention is XLA's typed FFI, ``api_version`` 4. jaxlib
0.11 writes neither onto the op. It leaves the op's own ``backend_config``
empty and its ``api_version`` at the original convention (1), parks the
dictionary in the *discardable* attribute ``mhlo.backend_config``, and relies
on a rewrite pattern inside XLA's StableHLO-to-HLO conversion to move the
dictionary back and re-tag the call as typed FFI. A discardable attribute has
no guarantee of reaching that pattern, and in large programs it sometimes
does not: the call then reaches XLA under the original convention, XLA looks
for an original-convention handler, finds none, and compilation fails with
``NOT_FOUND: No registered implementation for custom call to <target>``.

This module wraps :func:`jax._src.interpreters.mlir.custom_call`, the one
builder every JAX and diff_soup FFI lowering goes through, so that a
dictionary configuration is written as the op's inherent ``backend_config``
(a ``DictionaryAttr``) next to an inherent ``api_version = 4``, and no
discardable attribute takes part. StableHLO has accepted a dictionary
``backend_config`` since version 1.3.0, and XLA's exporter serializes it for
a typed-FFI call exactly as the rewrite pattern would have produced it, so
the HLO that XLA compiles, the numbers it computes, and its compile time are
those of a program on which the pattern did fire.

The wrapper installs itself when diff_soup is imported, provided this
jaxlib's ``custom_call`` still parks dictionaries in ``mhlo.backend_config``
(a jaxlib that writes them inherently needs no help) and the bundled
StableHLO is at least 1.3.0. Set ``DIFF_SOUP_INHERENT_FFI_CONFIG=0`` in the
environment before importing diff_soup to leave JAX's lowering untouched,
for instance to reproduce the failure or to compare lowered programs;
:func:`install` and :func:`uninstall` switch it within a process. Programs
lowered with the wrapper differ textually from programs lowered without it,
so the persistent compilation cache is filled afresh once.

One JAX component reads the discardable attribute back after lowering: the
Pallas Mosaic-GPU / PyTorch interop in ``jax._src.pallas.mosaic_gpu.torch``
(``attributes["mhlo.backend_config"]``). diff_soup never lowers through it;
a process that does should disable this wrapper.
"""

from __future__ import annotations

import functools
import inspect
import os
from typing import Any

from diff_soup._jax_compat import hlo, internal_mlir, ir

DISABLE_VARIABLE = "DIFF_SOUP_INHERENT_FFI_CONFIG"
"""Environment variable; the value ``0`` keeps the wrapper from installing."""

TYPED_FFI_API_VERSION = 4
"""XLA's ``CustomCallApiVersion::API_VERSION_TYPED_FFI``."""

MINIMUM_STABLEHLO_VERSION = (1, 3, 0)
"""First StableHLO whose ``custom_call`` serializes a dictionary ``backend_config``."""

_DISCARDABLE_CONFIG = "mhlo.backend_config"
_original_custom_call: Any = None


def stablehlo_version() -> tuple[int, ...]:
    """The StableHLO version bundled with the imported jaxlib, as a tuple."""
    return tuple(int(part) for part in hlo.get_current_version().split("."))


def jaxlib_parks_dictionary_configs() -> bool:
    """Whether this jaxlib's ``custom_call`` stores dictionaries in ``mhlo.backend_config``."""
    target = internal_mlir.custom_call if _original_custom_call is None else _original_custom_call
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        return False
    return _DISCARDABLE_CONFIG in source


def installed() -> bool:
    """Whether :func:`jax._src.interpreters.mlir.custom_call` is currently wrapped."""
    return _original_custom_call is not None


def install() -> bool:
    """Wrap JAX's custom-call builder; returns whether the wrapper is in place.

    Idempotent. Returns ``False``, leaving JAX untouched, when this jaxlib
    already writes dictionary configurations inherently or its StableHLO is
    older than 1.3.0.
    """
    global _original_custom_call
    if installed():
        return True
    if not jaxlib_parks_dictionary_configs() or stablehlo_version() < MINIMUM_STABLEHLO_VERSION:
        return False
    original = internal_mlir.custom_call

    @functools.wraps(original)
    def custom_call(call_target_name: str, *, backend_config: Any = "", **kwargs: Any):
        op = original(call_target_name, backend_config=backend_config, **kwargs)
        if isinstance(backend_config, dict):
            attributes = op.operation.attributes
            if _DISCARDABLE_CONFIG in attributes:
                config = attributes[_DISCARDABLE_CONFIG]
                del attributes[_DISCARDABLE_CONFIG]
            else:
                config = ir.DictAttr.get(backend_config)
            attributes["backend_config"] = config
            attributes["api_version"] = internal_mlir.i32_attr(TYPED_FFI_API_VERSION)
        return op

    _original_custom_call = original
    internal_mlir.custom_call = custom_call
    return True


def uninstall() -> None:
    """Restore JAX's own custom-call builder. A no-op when nothing is installed."""
    global _original_custom_call
    if installed():
        internal_mlir.custom_call = _original_custom_call
        _original_custom_call = None


if os.environ.get(DISABLE_VARIABLE, "1") != "0":
    install()
