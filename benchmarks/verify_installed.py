"""Run maintained release contracts against a wheel in an isolated environment."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import textwrap
import tomllib
import venv
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]

_FORBIDDEN_WHEEL_PARTS = {
    ".agents",
    ".claude",
    ".codex",
    "_dev",
    "examples",
    "reference-sources",
    "reference_sources",
    "tests",
}
# Checkout files the maintained suite reads besides ``tests/``: the benchmark
# schema and comparator, the docs budget tool, and the project metadata.
# ``examples/`` is copied only when the checkout has it.
_SUITE_SUPPORT_FILES = ("pyproject.toml",)
_SUITE_SUPPORT_DIRECTORIES = ("benchmarks", "examples", ".github/scripts")

# Packages the suite imports or exercises that the wheel does not depend on:
# the test runner and its timeout plugin, jsonschema for the benchmark schema
# rows, beartype for the runtime type-checking import hook, diffrax and
# optimistix for the Diffrax stepper rows, matplotlib for the pixel-averaging
# measurement rows.
_SUITE_REQUIREMENTS = (
    "pytest>=8",
    "pytest-timeout>=2",
    "jsonschema>=4",
    "beartype>=0.17",
    "diffrax>=0.7,<0.8",
    "optimistix>=0.0.10",
    "matplotlib>=3.5",
)

_PRIVATE_PLANNING_FILES = {
    "agents.md",
    "baton.md",
    "claude.md",
    "codex-prompt.md",
    "handoff.md",
}


def _declared_version() -> str:
    """Return the version the repository's pyproject.toml declares."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _audit_wheel_archive(wheel: Path) -> None:
    """Reject files outside the public package and wheel metadata before installation."""
    try:
        with zipfile.ZipFile(wheel) as archive:
            names = tuple(member.filename for member in archive.infolist() if member.filename)
    except (OSError, zipfile.BadZipFile) as error:
        raise SystemExit(f"wheel archive is unreadable: {wheel}") from error

    package_seen = False
    metadata_roots: set[str] = set()
    for name in names:
        if name.startswith("/") or "\\" in name:
            raise SystemExit(f"wheel archive contains an unsafe path: {name}")
        parts = PurePosixPath(name).parts
        lowered = tuple(part.lower() for part in parts)
        if not parts or ".." in parts:
            raise SystemExit(f"wheel archive contains an unsafe path: {name}")
        if any(
            part in _FORBIDDEN_WHEEL_PARTS
            or part.endswith(("-reference-sources", "_reference_sources"))
            for part in lowered
        ) or any(part in _PRIVATE_PLANNING_FILES for part in lowered):
            raise SystemExit(f"wheel archive contains private source material: {name}")
        root = parts[0]
        if root == "diff_soup":
            package_seen = True
        elif root.endswith(".dist-info"):
            metadata_roots.add(root)
        else:
            raise SystemExit(f"wheel archive contains an unexpected top-level path: {name}")
    if not package_seen or len(metadata_roots) != 1:
        raise SystemExit("wheel archive must contain diff_soup and exactly one .dist-info tree")
    print(f"passed: audited {len(names)} wheel archive members ({wheel.name})", flush=True)


def _surface_check(import_order: tuple[str, str]) -> str:
    """Return an isolated-import check for one public namespace order."""
    imports = "; ".join(f"importlib.import_module({name!r})" for name in import_order)
    return textwrap.dedent(
        f"""
        import importlib
        import importlib.metadata as metadata
        import sys
        import sysconfig
        from pathlib import Path

        {imports}
        import diff_soup

        site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
        loaded = tuple(
            (name, module)
            for name, module in sys.modules.items()
            if name == "diff_soup" or name.startswith("diff_soup.")
        )
        for module_name, module in loaded:
            module_file = getattr(module, "__file__", None)
            if module_file is None:
                continue
            module_path = Path(module_file).resolve()
            try:
                module_path.relative_to(site_packages)
            except ValueError as error:
                raise AssertionError(
                    f"{{module_name}} came from {{module_path}}, not {{site_packages}}"
                ) from error

        expected_version = {_declared_version()!r}
        assert metadata.version("diff_soup") == expected_version
        assert diff_soup.__version__ == expected_version
        required_exports = (
            "inference",
            "problem",
            "recipe",
            "DenseOutputPolicy",
            "AlgebraicProjectionOptions",
            "AlgebraicProjectionResult",
            "AlgebraicProjectionStatus",
            "project_algebraic",
            "ControlState",
            "OperationalControlConstraint",
            "FamilyGraph",
            "FamilyGraphExchange",
            "pdesolve",
            "SaveAt",
            "Solution",
            "JBDF",
            "JIDA",
            "JNDF",
            "ProblemSpec",
            "lower",
        )
        missing = tuple(
            name
            for name in required_exports
            if not hasattr(diff_soup, name) or name not in diff_soup.__all__
        )
        assert not missing, f"missing top-level exports: {{missing}}"

        from diff_soup.pdesolve import pdesolve
        from diff_soup.problem import ProblemSpec, lower
        from diff_soup.timestepping import SaveAt

        assert diff_soup.pdesolve is pdesolve
        assert diff_soup.SaveAt is SaveAt
        # ``diff_soup.problem`` is the problem package and the Recipe API verb:
        # calling it with a name starts a ProblemSpec builder.
        assert diff_soup.problem is sys.modules["diff_soup.problem"]
        assert callable(diff_soup.problem)
        assert isinstance(diff_soup.problem("surface_check"), ProblemSpec)
        assert diff_soup.inference is sys.modules["diff_soup.inference"]
        assert diff_soup.ProblemSpec is ProblemSpec
        assert diff_soup.lower is lower
        print(diff_soup.__file__)
        """
    )


def main() -> int:
    """Install one wheel away from the checkout and execute public/native gates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "gpu"))
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file():
        raise SystemExit(f"wheel does not exist: {wheel}")
    _audit_wheel_archive(wheel)

    with tempfile.TemporaryDirectory(prefix=f"diff-soup-{args.device}-wheel-suite-") as directory:
        clean_root = Path(directory)
        environment = clean_root / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = environment / "bin" / "python"
        requirement = f"diff-soup[cu12] @ {wheel.as_uri()}" if args.device == "gpu" else str(wheel)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                requirement,
                *_SUITE_REQUIREMENTS,
            ],
            check=True,
            cwd=clean_root,
        )
        shutil.copytree(
            ROOT / "tests",
            clean_root / "tests",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        for name in _SUITE_SUPPORT_FILES:
            shutil.copy2(ROOT / name, clean_root / name)
        for name in _SUITE_SUPPORT_DIRECTORIES:
            if (ROOT / name).is_dir():
                shutil.copytree(
                    ROOT / name,
                    clean_root / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
                )
        child_environment = os.environ.copy()
        if args.device == "cpu":
            child_environment["JAX_PLATFORMS"] = "cpu"
        else:
            child_environment.pop("JAX_PLATFORMS", None)

        # Import both public domains in fresh interpreters so module-cache order
        # cannot hide a circular dependency or a checkout-path leak.
        for import_order in (
            ("diff_soup.inference", "diff_soup.problem"),
            ("diff_soup.problem", "diff_soup.inference"),
        ):
            subprocess.run(
                [
                    str(python),
                    "-c",
                    _surface_check(import_order),
                ],
                check=True,
                cwd=clean_root,
                env=child_environment,
            )

        subprocess.run(
            [
                str(python),
                "-m",
                "pytest",
                "-q",
                "--import-mode=importlib",
                "--strict-markers",
                "--run-integration",
                "--run-validation",
                "tests/public_api",
                "tests/contracts",
                "tests/integration",
                "tests/validation",
            ],
            check=True,
            cwd=clean_root,
            env=child_environment,
        )
    print(f"passed: clean-installed {args.device} maintained suite ({wheel.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
