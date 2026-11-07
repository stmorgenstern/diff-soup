"""Deprecated: `diff_soup.inference` is now `diff_soup.optimization.inference`.

This package is an alias kept for one release and removed in 1.3. The first import warns
once with a `DeprecationWarning`. `import diff_soup.inference` returns the module
`diff_soup.optimization.inference` itself, and every submodule path under it resolves to
the module at its new home, so classes and caches are shared: nothing is imported twice.
`diff_soup.inference._abc` and `diff_soup.inference.solvers` resolve to
`diff_soup.optimization._abc` and `diff_soup.optimization.solvers`, where they moved.
"""

import importlib
import importlib.abc
import importlib.util
import sys

from diff_soup._warnings import warn_user

warn_user(
    "`diff_soup.inference` is deprecated and will be removed in diff-soup 1.3; use "
    "`diff_soup.optimization.inference` instead.",
    DeprecationWarning,
)

#: The old dotted prefixes and their new homes, the longest first.
_MOVED = (
    ("diff_soup.inference._abc", "diff_soup.optimization._abc"),
    ("diff_soup.inference.solvers", "diff_soup.optimization.solvers"),
    ("diff_soup.inference", "diff_soup.optimization.inference"),
)


def _new_name(name: str) -> str | None:
    """The new home of the old module `name`, or `None` when `name` is not under the alias."""
    for old, new in _MOVED:
        if name == old or name.startswith(old + "."):
            return new + name[len(old) :]
    return None


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import an old `diff_soup.inference.*` path as the module at its new home."""

    def find_spec(self, fullname, path=None, target=None):
        """A spec for an old submodule path, loaded by this finder; `None` for any other."""
        if fullname == __name__ or _new_name(fullname) is None:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        """The module at the new home, imported once under its own name."""
        module = importlib.import_module(_new_name(spec.name))
        spec.loader_state = module.__spec__
        return module

    def exec_module(self, module):
        """Nothing to run: the module ran under its new name. Give it back its own spec."""
        module.__spec__ = module.__spec__.loader_state


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

sys.modules[__name__] = importlib.import_module("diff_soup.optimization.inference")
