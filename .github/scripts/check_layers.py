"""Check that module-level imports inside ``diff_soup`` follow the layer order.

Every first-level name under ``diff_soup/`` (a subpackage or a module directly
in the package) belongs to one layer in :data:`LAYERS`, listed from the lowest
to the highest. A module may import, at module level, from its own layer or
from any layer below it. An import from a lower layer into a higher one is a
violation. Imports inside functions are not checked: they run on first call,
after every module has loaded, and are how the package breaks import cycles.

The violations that exist today are listed in ``.github/scripts/layers_baseline.txt``.
The check fails only on a violation that is not in that file, so the baseline
may shrink as imports are moved but never grow.

Usage::

    python .github/scripts/check_layers.py                  # fail on new violations
    python .github/scripts/check_layers.py --write-baseline # record the current tree
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

PACKAGE_ROOT: Path = (
    Path(__file__).resolve().parents[2] / "diff_soup"
)  # .github/scripts/ is two levels down
BASELINE_PATH: Path = Path(__file__).resolve().parent / "layers_baseline.txt"

PACKAGE = "diff_soup"

# The package's own ``__init__.py`` and ``from diff_soup import <attribute>``.
TOP = "__init__"

LAYERS: tuple[str, ...] = (
    # Leaves: import nothing from diff_soup, or only the leaves listed above them.
    "_version",  # reads the installed version; imports nothing from diff_soup
    "_jax_constants",  # sets JAX environment defaults before JAX loads; no imports
    "_jax_compat",  # the one place that touches JAX internals; imports only JAX
    "_jax_typed_ffi",  # typed FFI call helpers built on _jax_compat
    "_errors",  # the exception hierarchy every package raises
    "_optional",  # optional-dependency loader; raises an _errors type
    "_logging",  # the package logger; imports nothing from diff_soup
    "_warnings",  # warnings attributed to the user's line; imports nothing from diff_soup
    "_precision",  # dtype policy; raises _errors types
    "_layout",  # state-vector layout helpers; raises _errors types
    "_solution",  # solution container shared by solvers and the schedule
    "_adjoints",  # adjoint-mode markers shared by nonlinear and timestepping
    "_internal",  # native extension bindings; imported lazily by the solvers
    "utils",  # numerical helpers; imports only _jax_compat and _errors
    "_distributed",  # device-mesh helpers on _jax_compat; used by linear and inference
    "distributed",  # public launcher over _distributed
    # The solver stack.
    "linear",  # linear solvers; imports only the leaves
    "nonlinear",  # Newton solvers built on linear
    # The modeling spine: domains -> problem -> _lowering -> timestepping -> inference.
    "domains",  # meshes and geometry; the base of the modeling spine
    "problem",  # the Recipe API and residual IR; builds on linear and nonlinear
    "_lowering",  # turns a problem and its domains into solver-ready arrays
    "timestepping",  # time integrators over nonlinear and linear
    "pdesolve",  # the solve entry point; wires timestepping together
    "_compile",  # ds.compile: compiles one pdesolve call ahead of time
    "_schedule",  # multi-phase schedules over pdesolve; used by inference
    "inference",  # fitting and sensitivities over the whole solve stack
    # Top level: wires the packages together for the public namespace.
    "projection",  # consistent-initialization projection over the solver stack
    TOP,  # diff_soup/__init__.py, the public namespace
)


def _module_name(path: Path, package_root: Path) -> tuple[str, bool]:
    """Return the dotted module name of ``path`` and whether it is a package."""
    parts = list(path.relative_to(package_root.parent).with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _first_level(module: str) -> str:
    """Return the layer name of an absolute ``diff_soup`` module name."""
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else TOP


def _module_level_imports(body: list[ast.stmt]) -> Iterator[ast.Import | ast.ImportFrom]:
    """Yield the import statements that run when the module is imported."""
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, ast.If):
            yield from _module_level_imports(node.body)
            yield from _module_level_imports(node.orelse)
        elif isinstance(node, ast.Try):
            yield from _module_level_imports(node.body)
            for handler in node.handlers:
                yield from _module_level_imports(handler.body)
            yield from _module_level_imports(node.orelse)
            yield from _module_level_imports(node.finalbody)


def _imported_modules(
    node: ast.Import | ast.ImportFrom, module: str, is_package: bool, package_root: Path
) -> list[str]:
    """Return the absolute ``diff_soup`` modules one import statement names."""
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
        return [name for name in names if name == PACKAGE or name.startswith(PACKAGE + ".")]
    if node.level:
        base = module.split(".") if is_package else module.split(".")[:-1]
        base = base[: len(base) - (node.level - 1)]
        target = ".".join(base + ([node.module] if node.module else []))
    else:
        target = node.module or ""
    if target != PACKAGE:
        return [target] if target.startswith(PACKAGE + ".") else []
    # ``from diff_soup import x``: x is a first-level module or a top-level attribute.
    modules = []
    for alias in node.names:
        is_module = (package_root / alias.name).is_dir() or (
            package_root / f"{alias.name}.py"
        ).is_file()
        modules.append(f"{PACKAGE}.{alias.name}" if is_module else PACKAGE)
    return modules


def find_violations(package_root: Path = PACKAGE_ROOT) -> list[str]:
    """Find every module-level import that points from a lower layer to a higher one.

    Args:
        package_root: The ``diff_soup`` package directory to scan.

    Returns:
        Sorted, de-duplicated strings ``"importer -> imported"`` with absolute
        module names, the imported module as written in the import statement.

    Raises:
        ValueError: If a first-level name under the package is not in
            :data:`LAYERS`, or an import names a ``diff_soup`` module that does
            not exist.
    """
    rank = {name: index for index, name in enumerate(LAYERS)}
    unplaced = sorted(
        _first_level(_module_name(path, package_root)[0])
        for path in package_root.rglob("*.py")
        if _first_level(_module_name(path, package_root)[0]) not in rank
    )
    if unplaced:
        names = ", ".join(f"{PACKAGE}.{name}" for name in dict.fromkeys(unplaced))
        raise ValueError(
            f"{names} has no layer: add it to LAYERS in .github/scripts/check_layers.py "
            f"at the position its imports allow"
        )
    violations: set[str] = set()
    for path in sorted(package_root.rglob("*.py")):
        module, is_package = _module_name(path, package_root)
        source_layer = _first_level(module)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _module_level_imports(tree.body):
            for target in _imported_modules(node, module, is_package, package_root):
                target_layer = _first_level(target)
                if target_layer not in rank:
                    raise ValueError(
                        f"{module} imports {target}, which does not exist under {package_root}"
                    )
                if rank[target_layer] > rank[source_layer]:
                    violations.add(f"{module} -> {target}")
    return sorted(violations)


def load_baseline(path: Path = BASELINE_PATH) -> set[str]:
    """Read the recorded violations, one per line, skipping comments and blank lines.

    Args:
        path: The baseline file.

    Returns:
        The set of recorded violation strings.
    """
    entries = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return {entry for entry in entries if entry and not entry.startswith("#")}


def new_violations(
    package_root: Path = PACKAGE_ROOT, baseline_path: Path = BASELINE_PATH
) -> list[str]:
    """Return the violations in the tree that the baseline does not record.

    Args:
        package_root: The ``diff_soup`` package directory to scan.
        baseline_path: The baseline file.

    Returns:
        Sorted violation strings absent from the baseline.
    """
    baseline = load_baseline(baseline_path)
    return [violation for violation in find_violations(package_root) if violation not in baseline]


_BASELINE_HEADER = """\
# Module-level imports in diff_soup that point from a lower layer to a higher
# one (see LAYERS in .github/scripts/check_layers.py), recorded so the check fails only
# on new ones. Format: "importer -> imported", one per line.
#
# This list may only shrink. Remove an entry once its import is moved; do not
# add one: put the import in a function or move the code to the right layer.
"""


def main(argv: list[str] | None = None) -> int:
    """Run the layer check.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when ``None``.

    Returns:
        1 if the tree has a violation the baseline does not record, else 0.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="rewrite the baseline file from the current tree",
    )
    args = parser.parse_args(argv)

    try:
        violations = find_violations()
    except ValueError as error:
        print(error)
        return 1
    if args.write_baseline:
        BASELINE_PATH.write_text(
            _BASELINE_HEADER + "".join(f"{v}\n" for v in violations), encoding="utf-8"
        )
        print(f"wrote {len(violations)} violations to {BASELINE_PATH}")
        return 0

    baseline = load_baseline()
    new = [violation for violation in violations if violation not in baseline]
    if new:
        print(f"{len(new)} module-level imports point from a lower layer to a higher one:")
        for violation in new:
            print(f"  {violation}")
        print(
            "Import from the same layer or below "
            "(order: LAYERS in .github/scripts/check_layers.py),"
        )
        print("or move the import into the function that uses it.")
        return 1

    by_package = Counter(_first_level(violation.split(" -> ")[0]) for violation in violations)
    print(
        f"no new layer violations; {len(violations)} recorded in the baseline ({len(baseline)} entries)"
    )
    for package, count in sorted(by_package.items()):
        print(f"  {package}: {count}")
    stale = sorted(baseline - set(violations))
    if stale:
        print(
            f"{len(stale)} baseline entries no longer occur; remove them from {BASELINE_PATH.name}:"
        )
        for entry in stale:
            print(f"  {entry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
