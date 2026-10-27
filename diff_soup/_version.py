"""The installed package's version, read from its metadata.

A checkout on ``sys.path`` without an installed distribution reports the
version declared in its ``pyproject.toml``.
"""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path


def _read_version() -> str:
    try:
        return metadata.version("diff_soup")
    except metadata.PackageNotFoundError:
        pass
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
    except OSError:
        match = None
    return match.group(1) if match else "0.0.0"


__version__ = _read_version()

__all__ = ["__version__"]
