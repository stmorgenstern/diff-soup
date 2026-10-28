"""Import optional dependencies on first use.

Usage::

    from diff_soup._optional import import_optional

    gmsh = import_optional("gmsh")
"""

from __future__ import annotations

import importlib
from typing import Any

from diff_soup._errors import OptionalDependencyError

# The install command for every module diff_soup imports with ``import_optional``.
_INSTALL_HINTS: dict[str, str] = {
    "gmsh": 'pip install "diff-soup[mesh]"',
    # The default fallback ``pip install {name}`` would suggest the wrong
    # PyPI package name for ``cv2`` (the import is ``cv2`` but the package
    # is ``opencv-python``).
    "cv2": "pip install opencv-python",
    "meshio": "pip install \"diff-soup[mesh]\"",
    "optax": "pip install optax",
}


def import_optional(name: str) -> Any:
    """Import an optional dependency by module name.

    Args:
        name: The module to import, such as ``"gmsh"`` or ``"meshio"``.

    Returns:
        The imported module.

    Raises:
        OptionalDependencyError: If the module is not installed. It subclasses
            ``ImportError``.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        hint = _INSTALL_HINTS.get(name, f"pip install {name}")
        raise OptionalDependencyError(
            f"'{name}' is required for this feature but is not installed. Install it with: {hint}"
        ) from None
