"""Native cuDSS FFI loader: selects the extension built for the installed CUDA line.

Isolates the ``.so`` import wiring (CUDA-line detection, the NVIDIA-wheel
runtime preload, the cuDSS version check and the ``LD_LIBRARY_PATH`` prepend
for child processes) from the primitive-registration body, so the latter
reads as pure JAX glue.

The ``_native_solver`` extension is compiled once per CUDA line from one
source and lives in ``diff_soup/_internal/cudss_ffi/cu12/`` or ``cu13/``
(gitignored ``.so``; see the colocated ``CMakeLists.txt`` and "Installing
from source" in the installation guide). JAX runs on exactly one CUDA line,
chosen by the installed plugin package (``jax_cuda12_plugin`` or
``jax_cuda13_plugin``), and the extension must be built against that same
line: it links the CUDA runtime and cuDSS of its line, and those libraries
are not interchangeable across lines. This module detects the plugin, loads
the matching extension, and refuses a mismatch with an error naming both
sides.

The extension needs ``libcudart`` and ``libcudss`` at import, and cuDSS 1.x
needs cuBLAS and NVRTC when it is loaded.
Those ship in the NVIDIA pip wheels of the line (``nvidia-*-cu12``;
``nvidia-cudss-cu13`` plus the unsuffixed CUDA 13 wheels), whose ``lib/``
directories are not on the dynamic loader's default search path. The
extension does record the cuDSS and runtime directories as ``DT_RUNPATH``
when it is built, but that is only a fallback: the recorded directories need
not still exist, or hold the wheels actually installed, when the extension is
imported. This module therefore opens each library from its wheel with
``RTLD_GLOBAL`` before importing the extension, dependencies first, so the
extension's ``NEEDED`` entries and cuDSS's own resolve against the already
loaded libraries. A wheel that is not installed, or a library that does not
load, is passed over without a message; only when the extension then fails to
import does the error name it. ``LD_LIBRARY_PATH`` is
read once at process start, so editing it here cannot help the current
process; it is prepended only so child processes inherit the wheel
directories.

The extension records the cuDSS major version it was built against and
reports the version of the cuDSS library loaded in the process; a process
whose cuDSS has another major version is refused with a ``ConfigurationError``
naming both (not an ``ImportError``, so ``diff_soup.linear`` reports it
instead of treating the backend as absent).
The loader also hands the extension the path of cuDSS's threading layer
library from the cuDSS wheel, which ``DIFFSOUP_CUDSS_HOST_NTHREADS`` loads.
"""

from __future__ import annotations

import ctypes
import importlib
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import jax

from diff_soup._errors import ConfigurationError

# CUDA lines the extension is built for, keyed by the JAX plugin package that
# selects the line at runtime. Each entry lists the libraries the extension and
# cuDSS need at load time, in dependency order (the CUDA runtime, cuBLAS and
# NVRTC before cuDSS), as (importable package, file-name pattern in that
# package's ``lib/``, pip distribution) triples. cuDSS 1.x finds cuBLAS and
# NVRTC itself in the wheel layouts (its RPATH holds ``$ORIGIN`` and, on CUDA
# 12, ``$ORIGIN/../../cublas/lib`` and ``$ORIGIN/../../cuda_nvrtc/lib``); the
# preload is a convenience for other layouts, so a library it cannot find or
# load is passed over silently. The extension itself calls no cuBLAS or NVRTC
# routine.
_CUDA_LINES: dict[str, tuple[int, tuple[tuple[str, str, str], ...]]] = {
    "jax_cuda12_plugin": (
        12,
        (
            ("nvidia.cuda_runtime", "libcudart.so*", "nvidia-cuda-runtime-cu12"),
            ("nvidia.cublas", "libcublasLt.so*", "nvidia-cublas-cu12"),
            ("nvidia.cublas", "libcublas.so*", "nvidia-cublas-cu12"),
            ("nvidia.cuda_nvrtc", "libnvrtc.so*", "nvidia-cuda-nvrtc-cu12"),
            ("nvidia.cu12", "libcudss.so*", "nvidia-cudss-cu12"),
        ),
    ),
    "jax_cuda13_plugin": (
        13,
        (
            ("nvidia.cu13", "libcudart.so*", "nvidia-cuda-runtime"),
            ("nvidia.cu13", "libcublasLt.so*", "nvidia-cublas"),
            ("nvidia.cu13", "libcublas.so*", "nvidia-cublas"),
            # VERIFY: the CUDA 13 NVRTC wheel installs into nvidia/cu13/lib.
            ("nvidia.cu13", "libnvrtc.so*", "nvidia-cuda-nvrtc"),
            ("nvidia.cu13", "libcudss.so*", "nvidia-cudss-cu13"),
        ),
    ),
}

# The file-name pattern of cuDSS's threading layer library in the cuDSS wheel's
# ``lib/`` (the last entry of each line above names that wheel).
_THREADING_LAYER_PATTERN = "libcudss_mtlayer_gomp.so*"

# Handles of the libraries loaded below, held for the process lifetime so the
# loader never unmaps them under the extension.
_LOADED_RUNTIME_LIBRARIES: list[ctypes.CDLL] = []


def register_ffi_type_compat(name: str, type_id_capsule, type_info_capsule=None) -> None:
    """Register the state type of a stateful FFI handler with XLA.

    JAX 0.8 and later expose ``jax.ffi.register_ffi_type``, which takes the
    type id together with the type info (the deleter XLA calls to destroy the
    handler state with the executable; required since jaxlib 0.11). Earlier
    releases only have ``jax.ffi.register_ffi_type_id`` and take the id alone.

    Args:
        name: The FFI target name.
        type_id_capsule: The native type-id capsule from ``_native_solver``.
        type_info_capsule: The native type-info capsule from ``_native_solver``
            (``type_info_*``); omitted only on JAX releases that predate it.
    """
    if hasattr(jax.ffi, "register_ffi_type"):
        registration = {"type_id": type_id_capsule}
        if type_info_capsule is not None:
            registration["type_info"] = type_info_capsule
        jax.ffi.register_ffi_type(name, registration, platform="CUDA")
    else:
        jax.ffi.register_ffi_type_id(name, type_id_capsule, platform="CUDA")


def _package_installed(package: str) -> bool:
    """Whether an importable package of that name is installed."""
    try:
        return find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def _built_extension_lines() -> list[int]:
    """CUDA lines for which a ``_native_solver`` extension is built in this checkout."""
    root = Path(__file__).resolve().parents[2] / "_internal" / "cudss_ffi"
    return [line for line in (12, 13) if list((root / f"cu{line}").glob("_native_solver*.so"))]


def detect_cuda_line() -> str:
    """Return the JAX CUDA plugin package installed, which fixes the CUDA line.

    Returns:
        ``"jax_cuda12_plugin"`` or ``"jax_cuda13_plugin"``.

    Raises:
        ImportError: When no JAX CUDA plugin is installed, or when both are.
    """
    installed = [name for name in _CUDA_LINES if _package_installed(name)]
    if len(installed) == 1:
        return installed[0]
    if not installed:
        raise ImportError(
            "No JAX CUDA plugin is installed, so the native cuDSS extension cannot "
            "run. Install jax[cuda12] or jax[cuda13] (diff-soup[cu12] or "
            "diff-soup[cu13]) to use the cuDSS backend."
        )
    raise ImportError(
        "Both jax_cuda12_plugin and jax_cuda13_plugin are installed, so the CUDA "
        "line is ambiguous. Keep exactly one of jax[cuda12] and jax[cuda13]."
    )


def _wheel_lib_dir(package: str) -> Path | None:
    """Locate the ``lib/`` directory of an installed NVIDIA pip package.

    Args:
        package: Importable name of the NVIDIA package, e.g. ``nvidia.cublas``.

    Returns:
        The package's ``lib/`` directory, or ``None`` when the package is not
        installed or ships no such directory.
    """
    try:
        spec = find_spec(package)
    except (ImportError, ValueError):  # parent package absent or not a package
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    lib_dir = Path(spec.submodule_search_locations[0]) / "lib"
    return lib_dir if lib_dir.is_dir() else None


def _wheel_library(package: str, pattern: str) -> Path | None:
    """Locate one CUDA library inside an installed NVIDIA pip package.

    Args:
        package: Importable name of the NVIDIA package, e.g. ``nvidia.cublas``.
        pattern: Glob for the library file inside the package's ``lib/``
            directory, e.g. ``libcublas.so*``.

    Returns:
        The path to the library, preferring the shortest matching name (the
        bare soname over a fully versioned file, normally the same file
        through a symlink); ``None`` when the package or the library is absent.
    """
    lib_dir = _wheel_lib_dir(package)
    if lib_dir is None:
        return None
    matches = sorted(lib_dir.glob(pattern), key=lambda p: (len(p.name), p.name))
    return matches[0] if matches else None


def _load_cuda_runtime_libraries(libraries: tuple[tuple[str, str, str], ...]) -> list[str]:
    """Load the CUDA libraries the native cuDSS extension and cuDSS depend on.

    The extension and cuDSS name their dependencies by soname, so loading them
    here, from the NVIDIA pip packages, with global symbol visibility, before
    the extension itself is imported, lets the import succeed with no
    ``LD_LIBRARY_PATH`` set, and regardless of whether the library directories
    recorded when the extension was compiled still exist.

    Libraries the dynamic loader can already resolve (a system CUDA install, an
    ``LD_LIBRARY_PATH`` set before process start) are unaffected: the first
    definition of each soname wins, so the wheel copy is used only when nothing
    else provided it.

    Packages that are not installed, and libraries that fail to load, are
    skipped, and the extension import below reports any real failure; the
    returned notes name each skipped library for that report.

    Returns:
        One note per library that was not loaded: the pip distribution and
        the library pattern, and why (not installed, or the loader's error).
    """
    notes: list[str] = []
    for package, pattern, distribution in libraries:
        library = _wheel_library(package, pattern)
        if library is None:
            installed = _wheel_lib_dir(package) is not None
            notes.append(
                f"{distribution} holds no {pattern}"
                if installed
                else f"{distribution} is not installed ({pattern})"
            )
            continue
        try:
            handle = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:  # pragma: no cover - depends on the local CUDA install
            notes.append(f"{library.name} from {distribution} failed to load: {exc}")
            continue
        _LOADED_RUNTIME_LIBRARIES.append(handle)
    return notes


def _prepend_nvidia_wheel_libs(libraries: tuple[tuple[str, str, str], ...]) -> None:
    """Prepend the NVIDIA wheel ``lib/`` dirs to ``LD_LIBRARY_PATH``.

    This affects child processes only: the current process resolved its search
    path at start-up, and is served by the libraries loaded above instead.
    """
    lib_dirs: list[str] = []
    for package, _pattern, _distribution in libraries:
        lib_dir = _wheel_lib_dir(package)
        if lib_dir is not None and str(lib_dir) not in lib_dirs:
            lib_dirs.append(str(lib_dir))
    if not lib_dirs:
        return

    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    merged: list[str] = []
    for p in lib_dirs + existing:
        if p and p not in merged:
            merged.append(p)
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


def _build_hint(line: int) -> str:
    """How to build the CUDA ``line`` extension from a checkout, as a clause after "build it".

    The CUDA 13 nvcc wheel ships ``nvcc``; the CUDA 12 one does not, so a CUDA
    12 build names a toolkit's ``nvcc`` through ``DIFF_SOUP_NVCC``.
    """
    if line == 12:
        return (
            "with pip install --no-build-isolation --no-deps -C cmake.define.DIFF_SOUP_NVCC="
            '<a CUDA 12 nvcc> <checkout> after pip install "<checkout>[cu12]" '
            'scikit-build-core nanobind patchelf ("Installing from source" in the installation '
            "guide)"
        )
    return (
        "with pip install --no-build-isolation --no-deps <checkout> after pip install "
        f'"<checkout>[cu{line}]" nvidia-cuda-nvcc scikit-build-core nanobind patchelf '
        '("Installing from source" in the installation guide)'
    )


def _version_text(version: tuple[int, ...]) -> str:
    """``(1, 0, 0)`` as ``"1.0.0"``."""
    return ".".join(str(part) for part in version)


def check_cudss_version(module: Any, line: int) -> None:
    """Refuse an extension whose cuDSS major version differs from the loaded library's.

    Args:
        module: The imported extension, which reports the cuDSS version of
            its build (``cudss_built_version()``) and of the cuDSS library
            loaded in the process (``cudss_version()``), each as
            ``(major, minor, patch)``.
        line: The CUDA line of the extension, for the message.

    Raises:
        ConfigurationError: When the extension reports no cuDSS version (it
            was built against cuDSS 0.x), when the loaded library does not
            report its version, or when the two major versions differ.
    """
    built_version = getattr(module, "cudss_built_version", None)
    loaded_version = getattr(module, "cudss_version", None)
    if built_version is None or loaded_version is None:
        raise ConfigurationError(
            f"The native cuDSS extension for CUDA {line} reports no cuDSS version, so it was "
            f"built against cuDSS 0.x; diff_soup needs an extension built against cuDSS 1.x. "
            f"Rebuild it {_build_hint(line)}."
        )
    built = tuple(int(part) for part in built_version())
    try:
        loaded = tuple(int(part) for part in loaded_version())
    except RuntimeError as exc:
        raise ConfigurationError(
            f"The native cuDSS extension for CUDA {line} was built against cuDSS "
            f"{_version_text(built)}, and the cuDSS library loaded in this process does not "
            f"report its version ({exc}); install nvidia-cudss-cu{line}>={built[0]}.0.0,"
            f"<{built[0] + 1}."
        ) from exc
    if loaded[0] != built[0]:
        raise ConfigurationError(
            f"The native cuDSS extension for CUDA {line} was built against cuDSS "
            f"{_version_text(built)}, but the cuDSS library loaded in this process is "
            f"{_version_text(loaded)}; the extension runs only with cuDSS {built[0]}.x. "
            f"Install nvidia-cudss-cu{line}>={built[0]}.0.0,<{built[0] + 1} "
            f'(pip install "diff-soup[cu{line}]"), or rebuild the extension against the '
            f"installed cuDSS {_build_hint(line)}."
        )


def _register_threading_layer(module: Any, package: str) -> None:
    """Hand the extension the path of cuDSS's threading layer library in its wheel.

    Nothing is registered when the wheel ships no threading layer; the
    extension then refuses ``DIFFSOUP_CUDSS_HOST_NTHREADS`` unless
    ``CUDSS_THREADING_LIB`` names the library.
    """
    library = _wheel_library(package, _THREADING_LAYER_PATTERN)
    if library is not None:
        module.set_threading_layer(str(library))


def _import_native_solver():
    """Import the ``_native_solver`` extension built for the installed CUDA line."""
    plugin = detect_cuda_line()
    line, libraries = _CUDA_LINES[plugin]
    built = _built_extension_lines()
    if line not in built:
        others = ", ".join(f"CUDA {other}" for other in built) or "no CUDA line"
        raise ImportError(
            f"The installed JAX CUDA plugin is {plugin} (CUDA {line}), but the "
            f"native cuDSS extension in this checkout is built for {others}. "
            f"Build the CUDA {line} extension {_build_hint(line)}, or install the JAX "
            f"plugin of the line the extension was built for."
        )
    notes = _load_cuda_runtime_libraries(libraries)
    _prepend_nvidia_wheel_libs(libraries)
    try:
        module = importlib.import_module(f"diff_soup._internal.cudss_ffi.cu{line}._native_solver")
    except Exception as exc:  # pragma: no cover - import wiring only
        skipped = f" Not preloaded: {'; '.join(notes)}." if notes else ""
        wheels = ", ".join(sorted({distribution for _, _, distribution in libraries}))
        raise ImportError(
            f"The native cuDSS extension for CUDA {line} could not be imported ({exc})."
            f"{skipped} It needs the CUDA {line} NVIDIA wheels {wheels} (cuDSS 1.x); "
            f'install them with pip install "diff-soup[cu{line}]", or rebuild the extension '
            f"{_build_hint(line)}."
        ) from exc
    check_cudss_version(module, line)
    _register_threading_layer(module, libraries[-1][0])
    return module


native_solver = _import_native_solver()


__all__ = ["check_cudss_version", "detect_cuda_line", "native_solver", "register_ffi_type_compat"]
