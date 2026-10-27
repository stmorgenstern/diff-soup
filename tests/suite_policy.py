"""Enforce the small, contract-oriented shape of the test suite (see README.md)."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT / "tests"
MANIFEST_PATH = TEST_ROOT / "suite_manifest.json"
NON_TIER_DIRECTORIES = {"support", "__pycache__"}  # shared helpers, not a tier
#: The suite's own machinery: it probes capabilities, so the namespace rule exempts it.
INFRASTRUCTURE = {"tests/conftest.py", "tests/suite_policy.py"}
TIER_MEASURES = ("files", "functions", "lines")
#: Keyword arguments that state a numeric bound and therefore cite a measurement.
BOUND_KEYWORDS = {"atol", "rtol", "tol"}
#: How many lines above a bound its citing comment may sit.
CITATION_REACH = 3


def _ignored(path: Path) -> bool:
    """Caches and compiled files are not suite files."""
    parts = path.relative_to(TEST_ROOT).parts
    return (
        any(part == "__pycache__" or part.startswith(".") for part in parts)
        or path.suffix == ".pyc"
    )


class PolicyViolation(RuntimeError):
    """A test-suite change violates the declared review boundary."""


def _imported_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    base = node.module or ""
    return [base, *(f"{base}.{alias.name}" for alias in node.names)]


def _imported_diff_soup_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    """The ``diff_soup`` modules an import statement loads (not the names taken from them)."""
    if isinstance(node, ast.Import):
        modules = [alias.name for alias in node.names]
    else:
        modules = [node.module or ""]
    return [m for m in modules if m == "diff_soup" or m.startswith("diff_soup.")]


def _float_constant(node: ast.AST) -> float | None:
    """The value of a float literal (``1e-6`` or ``-1e-6``), else ``None``."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    if isinstance(node, ast.Constant) and isinstance(node.value, float):
        return node.value
    return None


def _cited(lines: list[str], lineno: int) -> bool:
    """A comment carrying the word ``measured`` sits on ``lineno`` or within reach above it."""
    for index in range(lineno, max(lineno - CITATION_REACH, 1) - 1, -1):
        text = lines[index - 1]
        if "#" in text and "measured" in text.split("#", 1)[1].lower():
            return True
    return False


def _check_bounds(path: Path, tree: ast.Module, lines: list[str]) -> None:
    """Every numeric bound carries a comment with its measured value (README rule 4)."""
    constants: dict[str, int] = {}  # module-level name -> line of its float literal
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and _float_constant(node.value) not in (None, 0.0):
                constants[target.id] = node.lineno

    def require(value: ast.AST, lineno: int, what: str) -> None:
        literal = _float_constant(value)
        if literal is not None:
            if literal != 0.0 and not _cited(lines, lineno):
                raise PolicyViolation(
                    f"{path}:{lineno}: {what} {literal!r} cites no measurement; add a comment "
                    "with the word 'measured', the value and the run (README rule 4)"
                )
        elif isinstance(value, ast.Name) and value.id in constants:
            where = constants[value.id]
            if not _cited(lines, where):
                raise PolicyViolation(
                    f"{path}:{where}: bound {value.id} cites no measurement; add a comment "
                    "with the word 'measured', the value and the run (README rule 4)"
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg in BOUND_KEYWORDS:
                    require(keyword.value, keyword.value.lineno, f"bound {keyword.arg}=")
        elif isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
            ordered = any(
                isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.test.ops
            )
            if ordered:
                for side in (node.test.left, *node.test.comparators):
                    require(side, node.lineno, "comparison bound")


def _check_source(
    path: Path,
    *,
    public: set[str],
    allowed_private: str | None,
    private_allowed_tier: bool,
    infrastructure: bool = False,
) -> tuple[int, int]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    test_functions = sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        for node in ast.walk(tree)
    )
    private_seen: set[str] = set()

    for node in ast.walk(tree):
        if (
            (isinstance(node, ast.Name) and node.id == "MonkeyPatch")
            or (isinstance(node, ast.Attribute) and node.attr == "MonkeyPatch")
            or (isinstance(node, ast.ImportFrom) and "MonkeyPatch" in {a.name for a in node.names})
        ):
            raise PolicyViolation(f"{path}: MonkeyPatch patches private seams and is banned")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = _imported_modules(node)
            if any(m == "mock" or m.startswith("mock.") or "unittest.mock" in m for m in modules):
                raise PolicyViolation(f"{path}: mocks are not test contracts")
            if "time" in modules or any(m.startswith("time.") for m in modules):
                raise PolicyViolation(f"{path}: wall-clock assertions belong in benchmarks")
            if any(m.startswith("tests.") and ".test_" in m for m in modules):
                raise PolicyViolation(f"{path}: tests may not import other test modules")
            for module in _imported_diff_soup_modules(node):
                if module in public or infrastructure:
                    continue
                if not private_allowed_tier:
                    raise PolicyViolation(
                        f"{path}: imports {module!r}, which is not a public namespace of the "
                        "manifest; only a contracts module may name one private module"
                    )
                owner = module if allowed_private is None else allowed_private
                if module != owner and not module.startswith(owner + "."):
                    raise PolicyViolation(
                        f"{path}: imports private module {module!r}; the manifest names "
                        f"{allowed_private!r} as this file's one private module"
                    )
                private_seen.add(owner)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fixture_names = {arg.arg for arg in (*node.args.args, *node.args.kwonlyargs)}
            forbidden = fixture_names & {"monkeypatch", "mocker"}
            if forbidden:
                raise PolicyViolation(
                    f"{path}: private-seam fixture {sorted(forbidden)[0]!r} is banned"
                )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner_name = node.func.value
            if isinstance(owner_name, ast.Name) and owner_name.id == "time":
                raise PolicyViolation(f"{path}: wall-clock assertions belong in benchmarks")
            if node.func.attr == "setattr" and isinstance(owner_name, ast.Name):
                if owner_name.id == "object":
                    raise PolicyViolation(f"{path}: object.__setattr__ writes a private seam")
    if private_allowed_tier and allowed_private is None and private_seen:
        raise PolicyViolation(
            f"{path}: imports private module {sorted(private_seen)[0]!r} without a "
            "private_imports entry in the manifest"
        )

    _check_bounds(path, tree, lines)
    return test_functions, len(lines)


def validate(*, allow_missing: bool = False) -> dict[str, dict[str, int]]:
    """Validate allowlists, budgets, tier ceilings and source rules; return per-tier sizes.

    Args:
        allow_missing: Report, instead of refuse, a listed module the tree does not hold yet.
    """
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    budgets = manifest["budgets"]
    expected = set(manifest["test_files"])
    expected_suite = set(manifest["suite_files"])
    public = set(manifest["public_namespaces"])
    private_imports = manifest["private_imports"]
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in TEST_ROOT.rglob("test_*.py")
        if not _ignored(path)
    }
    actual_suite = {
        path.relative_to(ROOT).as_posix()
        for path in TEST_ROOT.rglob("*")
        if path.is_file() and not _ignored(path)
    }

    unlisted = sorted(actual_suite - expected_suite)
    missing = sorted(expected_suite - actual_suite)
    if unlisted or (missing and not allow_missing):
        raise PolicyViolation(
            "test-suite file allowlist changed; reuse existing fixtures or make an explicit "
            f"manifest decision (unlisted={unlisted}, missing={missing})"
        )
    if missing:
        print(f"test-suite policy: {len(missing)} listed file(s) not in the tree yet: {missing}")
    if not expected <= set(manifest["suite_files"]):
        raise PolicyViolation("every test module must also be listed in suite_files")
    if len(actual) > budgets["test_files"]:
        raise PolicyViolation(f"test file budget exceeded: {len(actual)} > {budgets['test_files']}")
    for key in private_imports:
        if not key.startswith("tests/contracts/"):
            raise PolicyViolation(f"private_imports names a non-contracts module: {key}")

    stored_baselines = sorted(
        relative
        for relative in actual_suite
        if Path(relative).suffix.lower() in {".npy", ".npz", ".csv"}
        or (
            relative.startswith("tests/support/")
            and relative.endswith(".json")
            and relative not in manifest["support_data"]
        )
    )
    if stored_baselines:
        raise PolicyViolation(
            "stored baselines and snapshots are not portable analytical contracts "
            "(list a support data file in support_data): " + ", ".join(stored_baselines)
        )

    tiers = {
        path.name: dict.fromkeys(TIER_MEASURES, 0)
        for path in TEST_ROOT.iterdir()
        if path.is_dir() and path.name not in NON_TIER_DIRECTORIES
    }
    ceilings = manifest["tier_budgets"]
    if not set(tiers) <= set(ceilings):
        raise PolicyViolation(
            "every tier directory under tests/ needs a tier_budgets entry "
            f"(no ceiling={sorted(set(tiers) - set(ceilings))})"
        )
    for name in ceilings:
        tiers.setdefault(name, dict.fromkeys(TIER_MEASURES, 0))
    total = tiers["total"] = dict.fromkeys(TIER_MEASURES, 0)
    for relative in sorted(path for path in actual_suite if path.endswith(".py")):
        parts = relative.split("/")
        tier_name = parts[1] if len(parts) > 2 else None
        functions, lines = _check_source(
            ROOT / relative,
            public=public,
            allowed_private=private_imports.get(relative),
            private_allowed_tier=tier_name == "contracts",
            infrastructure=relative in INFRASTRUCTURE,
        )
        tier = tiers.get(tier_name) if tier_name else None
        for counts in (total, tier) if tier else (total,):
            counts["lines"] += lines
            if relative in actual:
                counts["files"] += 1
                counts["functions"] += functions

    if total["functions"] > budgets["test_functions"]:
        raise PolicyViolation(
            f"test function budget exceeded: {total['functions']} > {budgets['test_functions']}"
        )
    if total["lines"] > budgets["source_lines"]:
        raise PolicyViolation(
            f"test source budget exceeded: {total['lines']} > {budgets['source_lines']}"
        )
    for tier, ceiling in sorted(ceilings.items()):
        for key in TIER_MEASURES:
            if tiers[tier][key] > ceiling[key]:
                raise PolicyViolation(
                    f"{tier} tier {key} ceiling exceeded: {tiers[tier][key]} > {ceiling[key]}"
                )
    for key in ("files", "functions"):
        if (
            sum(ceiling[key] for ceiling in ceilings.values())
            > budgets[{"files": "test_files", "functions": "test_functions"}[key]]
        ):
            raise PolicyViolation(f"the tier ceilings for {key} exceed the suite budget")

    contracted_files = {row["file"] for row in manifest["feature_contracts"]}
    if contracted_files != expected:
        raise PolicyViolation(
            "every allowlisted test module must own one declared feature contract"
        )
    ceilings = dict(ceilings)
    ceilings["total"] = {
        "files": budgets["test_files"],
        "functions": budgets["test_functions"],
        "lines": budgets["source_lines"],
    }
    return {name: {k: (tiers[name][k], ceilings[name][k]) for k in TIER_MEASURES} for name in tiers}


def main(argv: list[str] | None = None) -> int:
    """Run the policy check; ``--report`` also prints each tier's actual/ceiling sizes."""
    args = sys.argv[1:] if argv is None else argv
    try:
        sizes = validate(allow_missing="--allow-missing" in args)
    except (OSError, SyntaxError, KeyError, json.JSONDecodeError, PolicyViolation) as exc:
        print(f"test-suite policy failed: {exc}", file=sys.stderr)
        return 1
    if "--report" in args:
        for tier in [*sorted(set(sizes) - {"total"}), "total"]:
            cells = "  ".join(f"{key} {a}/{c}" for key, (a, c) in sizes[tier].items())
            print(f"{tier + ':':<22} {cells}")
    print("test-suite policy passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
