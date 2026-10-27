"""The one import point for JAX modules and names outside JAX's stable API.

diff_soup defines its own primitives (with their lowering, batching and
differentiation rules), rewrites how JAX builds custom calls, and coordinates
the processes of a multi-host run. That takes modules JAX offers to extension
authors (``jax.extend``, ``jax.interpreters``), modules it marks experimental
(``jax.experimental``) and JAX's private implementation (``jax._src``). None
of them carries JAX's compatibility promise, so every module of diff_soup
takes them from here, and this module refuses to import under a JAX release
outside the range diff_soup supports (:data:`SUPPORTED_JAX`). The version is
checked before any of those modules is imported, so an unsupported JAX is
reported by version instead of by a missing internal name. The range is the
``jax`` requirement of diff_soup's package metadata (``pyproject.toml`` for a
checkout that is not installed), so the dependency pin is its one source.

The re-exports are the module objects themselves, not copies:
``internal_mlir`` is ``jax._src.interpreters.mlir`` (the module whose
``custom_call`` diff_soup wraps), ``mlir`` is the public
``jax.interpreters.mlir``, ``internal_partial_eval`` is
``jax._src.interpreters.partial_eval`` (whose rematerialisation policy types
the linear-solver primitives read in their partial-evaluation rules), and
``dispatch`` is ``jax._src.dispatch`` (which holds the tokens of the effects
of dispatched programs, waited on when a solve context is released).

This module imports JAX and nothing from diff_soup, so any module of the
package can import it without an import cycle. The package imports
``diff_soup._jax_constants`` before anything else, so JAX's constant-hoisting
setting is in place before this module first runs.
"""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

import jax

_RELEASE = re.compile(r"(\d+)\.(\d+)\.(\d+)(.*)")
_PRERELEASE = re.compile(r"[._-]?(a|b|c|rc|alpha|beta|pre|preview|dev)", re.IGNORECASE)
_JAX_REQUIREMENT = re.compile(r"jax\s*\(?\s*([<>=!~][^;()]*?)\s*\)?\s*(?:;.*)?$")
_CLAUSE = re.compile(r"(>=|<)\s*(\d+(?:\.\d+){0,2})")


def _jax_requirement() -> str:
    """Return the version specifier of diff_soup's ``jax`` requirement.

    Read from the installed package metadata; a checkout on ``sys.path``
    without an installed distribution reads ``[project].dependencies`` from
    its ``pyproject.toml``.
    """
    try:
        requirements = metadata.requires("diff_soup") or []
    except metadata.PackageNotFoundError:
        import tomllib

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with pyproject.open("rb") as handle:
            requirements = tomllib.load(handle)["project"]["dependencies"]
    for requirement in requirements:
        match = _JAX_REQUIREMENT.match(requirement.strip())
        if match is not None and "extra ==" not in requirement:
            return match.group(1)
    raise ImportError(
        "diff_soup's package metadata has no 'jax' requirement: declare "
        "'jax>=X.Y.Z,<X.Y' in [project].dependencies of pyproject.toml."
    )


def _release(version: str) -> tuple[int, int, int]:
    """Return ``version`` (one to three numeric parts) as a three-part release tuple."""
    major, minor, micro = ([int(part) for part in version.split(".")] + [0, 0])[:3]
    return major, minor, micro


def _supported_range(specifier: str) -> tuple[str, tuple[int, int, int], tuple[int, int, int]]:
    """Parse a ``>=minimum,<bound`` specifier into its text, minimum and exclusive bound.

    Raises:
        ImportError: If ``specifier`` is not exactly one ``>=`` and one ``<``
            clause on numeric releases.
    """
    clauses = {}
    for clause in specifier.split(","):
        match = _CLAUSE.fullmatch(clause.strip())
        if match is None or match.group(1) in clauses:
            clauses = None
            break
        clauses[match.group(1)] = match.group(2)
    if clauses is None or set(clauses) != {">=", "<"}:
        raise ImportError(
            f"diff_soup's 'jax' requirement is 'jax{specifier}'; it must be one '>=' "
            "and one '<' clause on numeric releases, such as 'jax>=0.11.1,<0.12'."
        )
    text = f">={clauses['>=']},<{clauses['<']}"
    return text, _release(clauses[">="]), _release(clauses["<"])


SUPPORTED_JAX, _MINIMUM, _BELOW = _supported_range(_jax_requirement())
"""The JAX releases diff_soup supports, as the requirement in its package metadata."""


def _check_jax_version(version: str) -> None:
    """Raise ``ImportError`` unless ``version`` lies in :data:`SUPPORTED_JAX`.

    A pre-release or development build of the minimum release lies below it.

    Args:
        version: A JAX version string, such as ``jax.__version__``.

    Raises:
        ImportError: If ``version`` is outside :data:`SUPPORTED_JAX` or is not a
            version string.
    """
    match = _RELEASE.match(version)
    if match is not None:
        release = tuple(int(part) for part in match.group(1, 2, 3))
        prerelease = _PRERELEASE.match(match.group(4)) is not None
        if _MINIMUM <= release < _BELOW and not (release == _MINIMUM and prerelease):
            return
    raise ImportError(
        f"diff_soup requires jax{SUPPORTED_JAX}, found jax {version}: install a jax "
        "release in that range."
    )


_check_jax_version(jax.__version__)

from jax._src import dispatch  # noqa: E402
from jax._src.interpreters import mlir as internal_mlir  # noqa: E402
from jax._src.interpreters import partial_eval as internal_partial_eval  # noqa: E402
from jax._src.lib.mlir import ir  # noqa: E402
from jax._src.lib.mlir.dialects import hlo  # noqa: E402
from jax.experimental import mesh_utils, multihost_utils  # noqa: E402
from jax.extend.core import Primitive, Var, jaxpr_as_fun  # noqa: E402
from jax.interpreters import ad, batching, mlir, partial_eval  # noqa: E402

__all__ = [
    # jax._src: JAX's private implementation (``internal_partial_eval`` holds
    # the rematerialisation policy types a partial-evaluation rule reads;
    # ``dispatch`` holds the tokens of the effects of dispatched programs)
    "dispatch",
    "internal_partial_eval",
    "internal_mlir",
    "ir",
    "hlo",
    # jax.experimental: device meshes and multi-process collectives
    "mesh_utils",
    "multihost_utils",
    # jax.extend.core: defining primitives
    "Primitive",
    "Var",
    "jaxpr_as_fun",
    # jax.interpreters: the transformation rules of primitives
    "ad",
    "batching",
    "mlir",
    "partial_eval",
]
