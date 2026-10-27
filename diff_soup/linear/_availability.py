"""Optional-backend availability probe for the cuDSS native extension.

The cuDSS backend (``diff_soup.linear._cudss``) links the GPU-only
``_native_solver`` FFI extension; on a CPU host (or any host without the built
``.so`` / NVIDIA runtime wheels) that import fails and the package degrades to
the CPU KLU/Dense path. [cudss_available][] is the single
host-side source of truth for "is the cuDSS backend usable here", mirroring
[klu_available][diff_soup.linear.klu_backend.klu_available].

It is deliberately a *lazy* guarded import (the native module is imported
INSIDE the function, never at module load) so importing this module is free on
a CPU host and never triggers a CUDA dlopen.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def cudss_available() -> bool:
    """Return ``True`` iff the native cuDSS backend is importable on this host.

    Caches the first probe — the answer is fixed for the process lifetime (the
    native extension is either present or it is not). Callers should prefer this
    over inspecting the ``diff_soup.linear.CuDSS is None`` sentinel directly.

    Returns:
        ``True`` when ``diff_soup.linear._cudss._backend.CuDSS`` imports and is
        non-``None`` (GPU host with the built extension + NVIDIA runtime libs);
        ``False`` otherwise.
    """
    try:
        from diff_soup.linear._cudss._backend import CuDSS
    except Exception:  # noqa: BLE001 — native ext / GPU / runtime libs absent
        return False
    return CuDSS is not None
