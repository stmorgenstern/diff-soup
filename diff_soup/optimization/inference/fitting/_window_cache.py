"""Content-addressed store of decision-variable window initializations.

Building a decision-variable multiple-shooting problem runs a forward sweep
over the horizon to place every window's start on the constraint manifold.
The sweep depends on the problem's structure, its parameter values at the
start (fitted and fixed), the initial state, the observation times and window layout, the
window-state basis, the first-step policy, the solver, and its numeric
policy, and on nothing else, so its result is stored under a key made of
exactly those inputs and read back before any later construction sweeps.

The store lives in the library's cache directory: ``DIFFSOUP_CACHE_DIR`` when
set, otherwise a ``diff_soup`` directory next to JAX's persistent compilation
cache when that is configured, otherwise ``~/.cache/diff_soup``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import jax
import numpy as np

from diff_soup.inference.fitting.windows import WindowInitialization
from diff_soup.problem.implicit.solver_problem import get_or_build_solver_view

_SUBDIRECTORY = "window-initialization"


def cache_directory() -> Path:
    """The directory the library persists its records under.

    A multi-process run reads records other processes wrote, so the
    directory must be shared: with several processes and neither
    ``DIFFSOUP_CACHE_DIR`` nor a persistent compilation cache configured, the
    per-user default is refused.
    """
    override = os.environ.get("DIFFSOUP_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    jax_cache = jax.config.jax_compilation_cache_dir
    if jax_cache:
        return Path(jax_cache).expanduser().parent / "diff_soup"
    if jax.process_count() > 1:
        raise RuntimeError(
            "A multi-process run stores window initializations in a directory every "
            "process can read, and the default ~/.cache/diff_soup is not shared. Set "
            "DIFFSOUP_CACHE_DIR to a shared directory, or JAX_COMPILATION_CACHE_DIR, "
            "whose parent then holds the store."
        )
    return Path("~/.cache/diff_soup").expanduser()


def window_initialization_key(
    *,
    seam: Any,
    y0: Any,
    times: Any,
    boundaries: Any,
    states: Any,
    restart_step: Any,
    solver: Any,
    project_windows: bool,
) -> str:
    """The content key of one window-initialization sweep."""
    view = get_or_build_solver_view(seam.problem)
    signature = view.structural_signature
    # A ``content`` model key is a digest of the problem's values and code, so
    # it is stable across processes and tells apart two authored problems that
    # differ only in a captured constant; an ``identity`` key is an object id
    # and a ``declared`` key holds layout and precision objects the encoding
    # below does not cover, so neither enters.
    structure = tuple(
        (field.name, getattr(signature, field.name))
        for field in dataclasses.fields(signature)
        if field.name != "source_key" or _is_content_key(signature.source_key)
    )
    modes = getattr(states, "modes", None)
    # Parameter values that are not fitted still decide the sweep (a fixed
    # input, a grid's size), and the structure does not record them.
    frozen = tuple(np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(seam.theta_frozen))
    declaration = (
        "window-initialization-v1",
        structure,
        np.asarray(seam.x0, dtype=float),
        frozen,
        np.asarray(y0, dtype=float),
        np.asarray(times, dtype=float),
        np.asarray(boundaries, dtype=float),
        None if modes is None else np.asarray(modes, dtype=float),
        None
        if restart_step is None
        else (
            str(restart_step.mode),
            float(restart_step.value),
            float(restart_step.min_step),
            float(restart_step.max_step),
            None if restart_step.hints is None else np.asarray(restart_step.hints, dtype=float),
        ),
        (
            type(solver).__name__,
            float(getattr(solver, "rtol", 0.0)),
            float(getattr(solver, "atol", 0.0)),
            str(getattr(solver, "linear", "")),
            repr(getattr(solver, "precision", None)),
            str(getattr(solver, "ic_mode", "")),
        ),
        bool(project_windows),
    )
    digest = hashlib.sha256()
    _feed(digest, declaration)
    return digest.hexdigest()


def _is_content_key(source_key: Any) -> bool:
    """Whether a structure signature's model key is a ``("content", digest)`` key."""
    return isinstance(source_key, tuple) and bool(source_key) and source_key[0] == "content"


def _feed(digest: Any, value: Any) -> None:
    """Feed one value into ``digest`` in a form that depends on its content alone."""
    if value is None:
        digest.update(b"N")
    elif isinstance(value, np.ndarray):
        digest.update(f"A{value.shape}{value.dtype}".encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, bytes):
        digest.update(b"B" + value)
    elif isinstance(value, (bool, int, float, str, np.generic)):
        digest.update(f"S{type(value).__name__}:{value!r};".encode())
    elif isinstance(value, (tuple, list)):
        digest.update(b"(")
        for item in value:
            _feed(digest, item)
        digest.update(b")")
    elif isinstance(value, dict):
        digest.update(b"{")
        for key in sorted(value, key=repr):
            _feed(digest, key)
            _feed(digest, value[key])
        digest.update(b"}")
    elif dataclasses.is_dataclass(value):
        _feed(digest, tuple((f.name, getattr(value, f.name)) for f in dataclasses.fields(value)))
    else:
        raise TypeError(
            "A window-initialization key can encode arrays, numbers, strings, bytes, "
            "tuples, dicts, and dataclasses; the problem's structure carries a "
            f"{type(value).__name__}."
        )


def _record_path(key: str) -> Path:
    return cache_directory() / _SUBDIRECTORY / f"{key}.npz"


def load_window_initialization(key: str) -> WindowInitialization | None:
    """The stored initialization under ``key``, or ``None`` when none is stored."""
    path = _record_path(key)
    if not path.is_file():
        return None
    with np.load(path) as data:
        coordinates = np.asarray(data["initial_coordinates"])
        return WindowInitialization(
            anchors=np.asarray(data["anchors"]),
            initial_coordinates=None if coordinates.size == 0 else coordinates,
            reached_tend=bool(data["reached_tend"]),
            provenance={"content_key": key, "path": str(path)},
        )


def store_window_initialization(key: str, initialization: WindowInitialization) -> Path:
    """Write ``initialization`` under ``key`` atomically and return its path."""
    path = _record_path(key)
    coordinates = initialization.initial_coordinates
    if not np.all(np.isfinite(np.asarray(initialization.anchors, dtype=float))) or (
        coordinates is not None and not np.all(np.isfinite(np.asarray(coordinates, dtype=float)))
    ):
        raise ValueError(
            "A window initialization with non-finite anchors or coordinates cannot be "
            "stored; the sweep that produced it failed."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchors": np.asarray(initialization.anchors, dtype=float),
        "initial_coordinates": (
            np.zeros((0,)) if coordinates is None else np.asarray(coordinates, dtype=float)
        ),
        "reached_tend": np.asarray(bool(initialization.reached_tend)),
    }
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f"{key}.", suffix=".part")
    os.close(handle)
    try:
        with open(temporary, "wb") as stream:
            np.savez(stream, **payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


__all__ = [
    "cache_directory",
    "load_window_initialization",
    "store_window_initialization",
    "window_initialization_key",
]
