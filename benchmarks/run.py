"""Run versioned diff_soup benchmark suites and emit machine-readable evidence."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import statistics
import subprocess
import tempfile
import time
import venv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks import SCHEMA_VERSION, SUITE_VERSION

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "benchmarks" / "schema.json"
jax: Any = None
jax_backend: Any = None
np: Any = None
select: Any = None


def _load_runtime() -> None:
    """Load numerical dependencies only in a process that executes benchmarks."""
    global jax, jax_backend, np, select
    if jax is not None:
        return
    # diff_soup sets JAX's constant-hoisting option, which takes full effect
    # only when it is set before jax is first imported.
    import diff_soup  # noqa: F401

    # isort: split
    import jax as loaded_jax
    import numpy as loaded_numpy
    from jax.extend import backend as loaded_jax_backend

    from benchmarks.cases import select as loaded_select

    jax = loaded_jax
    jax_backend = loaded_jax_backend
    np = loaded_numpy
    select = loaded_select


def _version(distribution: str) -> str | None:
    """Return an installed distribution version when it is available."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_text(command: list[str]) -> str | None:
    """Run a short diagnostic command and return its stripped stdout."""
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _required_text(command: list[str], *, description: str, allow_empty: bool = False) -> str:
    """Run a source-provenance command and fail closed if it does not succeed."""
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"cannot determine source {description}: Git command failed") from error
    value = completed.stdout.strip()
    if not value and not allow_empty:
        raise SystemExit(f"cannot determine source {description}: Git returned no value")
    return value


def _validate_commit(commit: str, *, qualifier: str) -> str:
    """Require one full lowercase Git object name."""
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise SystemExit(f"{qualifier} must be a full lowercase Git commit")
    return commit


def _git_source_state(root: Path = ROOT) -> tuple[str, bool]:
    """Read the exact source commit and dirty bit through mandatory Git commands."""
    commit = _required_text(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
        description="Git commit",
    )
    status = _required_text(
        ["git", "-C", str(root), "status", "--porcelain"],
        description="Git status",
        allow_empty=True,
    )
    return _validate_commit(commit, qualifier="source Git commit"), bool(status)


def _memory_gib() -> float | None:
    """Return total host memory in GiB when the platform exposes it."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / 1024**3, 3)
    except (OSError, ValueError):
        return None


def _gpu_identity() -> dict[str, Any]:
    """Collect stable GPU identity fields through ``nvidia-smi``."""
    query = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not query:
        return {"model": None, "count": 0, "memory_mib": None, "driver": None}
    rows = [row.split(", ") for row in query.splitlines()]
    return {
        "model": rows[0][0],
        "count": len(rows),
        "memory_mib": int(float(rows[0][1])),
        "driver": rows[0][2],
    }


def _environment() -> dict[str, Any]:
    """Describe the compatibility-relevant benchmark environment."""
    devices = jax.devices()
    gpu = _gpu_identity()
    backend = jax.default_backend()
    platform_version = getattr(jax_backend.get_backend(), "platform_version", "")
    flags = {
        name: os.environ[name]
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "JAX_PLATFORMS",
            "JAX_COMPILATION_CACHE_DIR",
            "XLA_FLAGS",
            "XLA_PYTHON_CLIENT_ALLOCATOR",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
        )
        if name in os.environ
    }
    return {
        "python": platform.python_version(),
        "jax": _version("jax"),
        "jaxlib": _version("jaxlib"),
        "xla": platform_version or None,
        "numpy": np.__version__,
        "cuda": platform_version if "cuda" in platform_version.lower() else None,
        "driver": gpu["driver"],
        "cudss": _version("nvidia-cudss-cu12") or _version("nvidia-cudss-cu13"),
        "klu": "SuiteSparse-7.10.1" if _version("diff_soup") else None,
        "compiler": platform.python_compiler(),
        "os": f"{platform.system()} {platform.release()}",
        "cpu_model": platform.processor() or _run_text(["uname", "-m"]) or "unknown",
        "cpu_count": os.cpu_count(),
        "ram_gib": _memory_gib(),
        "gpu_model": gpu["model"],
        "gpu_count": gpu["count"],
        "gpu_memory_mib": gpu["memory_mib"],
        "runner": os.getenv("RUNNER_NAME") or os.getenv("SLURM_JOB_ID") or platform.node(),
        "flags": flags,
        "x64": bool(jax.config.x64_enabled),
        "dtype": "float64" if jax.config.x64_enabled else "float32",
        "precision": "default",
        "device": backend,
        "device_count": len(devices),
    }


def _environment_key(environment: dict[str, Any]) -> str:
    """Hash compatibility fields into an exact-comparison environment key."""
    fields = (
        "python",
        "jax",
        "jaxlib",
        "xla",
        "numpy",
        "cuda",
        "driver",
        "cudss",
        "klu",
        "compiler",
        "os",
        "cpu_model",
        "cpu_count",
        "ram_gib",
        "gpu_model",
        "gpu_count",
        "gpu_memory_mib",
        "flags",
        "x64",
        "dtype",
        "precision",
        "device",
        "device_count",
    )
    identity = {field: environment[field] for field in fields}
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return f"{environment['device']}-{digest}"


def _file_hash(path: Path) -> str:
    """Compute a SHA-256 digest for a release artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(wheel: Path | None) -> dict[str, Any]:
    """Record source or installed-wheel provenance for one result."""
    import diff_soup

    inherited_commit = os.getenv("DIFF_SOUP_QUALIFICATION_COMMIT")
    inherited_dirty = os.getenv("DIFF_SOUP_QUALIFICATION_DIRTY")
    if inherited_dirty not in (None, "0", "1"):
        raise SystemExit("DIFF_SOUP_QUALIFICATION_DIRTY must be 0 or 1")
    if inherited_commit is not None:
        inherited_commit = _validate_commit(
            inherited_commit, qualifier="DIFF_SOUP_QUALIFICATION_COMMIT"
        )
    if wheel is not None:
        if inherited_commit is None or inherited_dirty is None:
            raise SystemExit(
                "installed-wheel qualification requires validated commit and dirty qualifiers"
            )
        commit, dirty = inherited_commit, inherited_dirty == "1"
    else:
        commit, dirty = _git_source_state()
        if inherited_commit is not None and inherited_commit != commit:
            raise SystemExit("DIFF_SOUP_QUALIFICATION_COMMIT does not match the source Git commit")
        if inherited_dirty is not None and (inherited_dirty == "1") != dirty:
            raise SystemExit("DIFF_SOUP_QUALIFICATION_DIRTY does not match the source Git status")
    imported = Path(diff_soup.__file__).resolve()
    source_package = (ROOT / "diff_soup").resolve()
    source_check = "wheel" if wheel and source_package not in imported.parents else "source"
    return {
        "git_commit": commit,
        "dirty": dirty,
        "wheel_filename": wheel.name if wheel else None,
        "wheel_sha256": _file_hash(wheel) if wheel else None,
        "imported_from": str(imported),
        "source_tree_import_check": source_check,
    }


def _block(value: Any) -> Any:
    """Synchronize every device-backed leaf before timing completes."""
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _jaxpr_equations(closed_jaxpr) -> list[Any]:
    """Flatten equations from a closed jaxpr and all nested jaxprs."""
    equations: list[Any] = []

    def visit(value):
        # A closed jaxpr wraps its jaxpr; a jaxpr's own ``jaxpr`` attribute
        # refers back to itself, so only descend when it is a different object.
        inner = getattr(value, "jaxpr", None)
        if inner is not None and inner is not value:
            visit(inner)
            return
        if hasattr(value, "eqns"):
            for equation in value.eqns:
                equations.append(equation)
                for parameter in equation.params.values():
                    visit(parameter)
            return
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(closed_jaxpr)
    return equations


def _scan_signature(equation) -> tuple:
    """Describe a scan by features that do not depend on the batch width.

    The signature is the scan length and direction, the dtype and rank of every
    operand and result, and the primitive sequence of the scan body. Batching
    adds a leading axis to the same operands at every width, so a solver-internal
    scan has the same signature at width 1 as at any other width.
    """

    def avals(variables):
        return tuple((str(v.aval.dtype), len(v.aval.shape)) for v in variables)

    body = equation.params["jaxpr"]
    body = getattr(body, "jaxpr", body)
    return (
        int(equation.params.get("length", -1)),
        bool(equation.params.get("reverse", False)),
        avals(equation.invars),
        avals(equation.outvars),
        tuple(inner.primitive.name for inner in body.eqns),
    )


_SINGLE_LANE_SCANS: dict[str, Counter | None] = {}


def _single_lane_scans(case) -> Counter | None:
    """Return the scan signatures of ``case`` traced at batch width 1, or ``None``.

    The trace is cached per case. It only builds a jaxpr, never compiles or runs
    it, and it is placed on the host CPU device when one is available. ``None``
    means the case declares no width-1 build or tracing it failed.
    """
    if case.name in _SINGLE_LANE_SCANS:
        return _SINGLE_LANE_SCANS[case.name]
    signatures = None
    if case.single_lane_build is not None:
        try:
            host = jax.default_device(jax.devices("cpu")[0])
        except RuntimeError:
            host = contextlib.nullcontext()
        for placement in (host, contextlib.nullcontext()):
            try:
                with placement:
                    workload = case.single_lane_build()
                    closed_jaxpr = jax.make_jaxpr(workload.function)(*workload.arguments)
            except Exception:  # noqa: BLE001
                continue
            signatures = Counter(
                _scan_signature(equation)
                for equation in _jaxpr_equations(closed_jaxpr)
                if equation.primitive.name == "scan"
            )
            break
    _SINGLE_LANE_SCANS[case.name] = signatures
    return signatures


def _structure(case, closed_jaxpr, stablehlo: str) -> dict[str, Any]:
    """Evaluate retained no-serialization and custom-call assertions.

    A scan whose length equals the batch width or direction count, and which
    has scanned inputs, counts as a batch-lane scan unless the width-1 trace of
    the same case carries a scan with the same signature: such a scan belongs
    to the solver and its length only coincides with the batch width. Cases
    without a width-1 trace count every such scan.
    """
    equations = _jaxpr_equations(closed_jaxpr)
    batch_lengths = {value for value in (case.batch_width, case.direction_count) if value > 1}
    candidates = [
        equation
        for equation in equations
        if equation.primitive.name == "scan"
        and int(equation.params.get("length", -1)) in batch_lengths
        and len(equation.invars)
        > int(equation.params.get("num_consts", 0)) + int(equation.params.get("num_carry", 0))
    ]
    reference = _single_lane_scans(case) if candidates else None
    remaining = Counter(reference or ())
    lane_scans = []
    for equation in candidates:
        signature = _scan_signature(equation)
        if remaining[signature] > 0:
            remaining[signature] -= 1
        else:
            lane_scans.append(int(equation.params.get("length", -1)))
    maps = sum(equation.primitive.name == "map" for equation in equations)
    expected = {target: (target in stablehlo) for target in case.expected_targets}
    forbidden = {target: (target in stablehlo) for target in case.forbidden_targets}
    passed = not lane_scans and maps == 0 and all(expected.values()) and not any(forbidden.values())
    return {
        "passed": passed,
        "batch_lane_scan_lengths": lane_scans,
        "single_lane_reference": (reference is not None) if candidates else None,
        "map_primitive_count": maps,
        "expected_targets": expected,
        "forbidden_targets_present": forbidden,
    }


def _device_peak_bytes() -> int | None:
    """Return the largest reported peak allocation across JAX devices."""
    peaks = []
    for device in jax.devices():
        try:
            stats = device.memory_stats()
        except Exception:  # noqa: BLE001
            stats = None
        if stats and stats.get("peak_bytes_in_use") is not None:
            peaks.append(int(stats["peak_bytes_in_use"]))
    return max(peaks) if peaks else None


def _measure(case, warmups: int, samples: int) -> dict[str, Any]:
    """Compile, execute, validate, and serialize one benchmark case."""
    metadata = case.metadata()
    try:
        workload = case.build()
        trace_start = time.perf_counter_ns()
        closed_jaxpr = jax.make_jaxpr(workload.function)(*workload.arguments)
        trace_ms = (time.perf_counter_ns() - trace_start) / 1e6

        compiled_function = jax.jit(workload.function)
        lower_start = time.perf_counter_ns()
        lowered = compiled_function.lower(*workload.arguments)
        lower_ms = (time.perf_counter_ns() - lower_start) / 1e6
        stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))

        compile_start = time.perf_counter_ns()
        executable = lowered.compile()
        compile_ms = (time.perf_counter_ns() - compile_start) / 1e6

        for _ in range(warmups):
            _block(executable(*workload.arguments))
        raw = []
        output = None
        for _ in range(samples):
            started = time.perf_counter_ns()
            output = _block(executable(*workload.arguments))
            raw.append((time.perf_counter_ns() - started) / 1e6)

        assert output is not None
        correct, error, tolerance = workload.check(output)
        structure = _structure(case, closed_jaxpr, stablehlo)
        telemetry = workload.telemetry()
        serialized = int(
            telemetry.get(
                "native_serialized_fallback_calls",
                telemetry.get("serialized_batch_calls", 0),
            )
        )
        passed = bool(correct and structure["passed"] and serialized == 0)
        median = statistics.median(raw)
        mad = statistics.median(abs(value - median) for value in raw)
        p95 = float(np.percentile(np.asarray(raw), 95))
        fingerprint = hashlib.sha256(stablehlo.encode()).hexdigest()
        evidence_dir = Path(os.getenv("DIFF_SOUP_BENCHMARK_EVIDENCE", "benchmark-results/evidence"))
        evidence_reference = f"{case.name}-{fingerprint[:12]}.stablehlo.txt"
        if os.getenv("DIFF_SOUP_WRITE_BENCHMARK_EVIDENCE") == "1":
            evidence_dir.mkdir(parents=True, exist_ok=True)
            (evidence_dir / evidence_reference).write_text(stablehlo, encoding="utf-8")
        return {
            **metadata,
            "status": "passed" if passed else "failed",
            "error": None,
            "measurements": {
                "trace_ms": trace_ms,
                "lower_ms": lower_ms,
                "compile_ms": compile_ms,
                "warmups": warmups,
                "raw_ms": raw,
                "median_ms": median,
                "mad_ms": mad,
                "p95_ms": p95,
                "peak_host_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                "peak_device_bytes": _device_peak_bytes(),
            },
            "solver_stats": {
                "accepted_steps": telemetry.get("accepted_steps"),
                "rejected_steps": telemetry.get("rejected_steps"),
                "nonlinear_iterations": telemetry.get("nonlinear_iterations"),
                "factorizations": telemetry.get(
                    "factorizations", telemetry.get("numeric_factorizations")
                ),
                "solve_calls": telemetry.get("solve_calls"),
            },
            "native_telemetry": telemetry,
            "correctness": {
                "passed": bool(correct),
                "error": float(error),
                "tolerance": float(tolerance),
            },
            "structure": structure,
            "evidence": {
                "stablehlo_sha256": fingerprint,
                "reference": evidence_reference,
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **metadata,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "measurements": None,
            "solver_stats": {},
            "native_telemetry": {},
            "correctness": {"passed": False, "error": None, "tolerance": None},
            "structure": {"passed": False},
            "evidence": {"stablehlo_sha256": None, "reference": None},
        }


def _validate(result: dict[str, Any]) -> None:
    """Validate a benchmark result against the versioned JSON schema."""
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("benchmark result validation requires jsonschema>=4") from exc
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)


def _run_suite(args, wheel: Path | None) -> dict[str, Any]:
    """Execute a selected suite and assemble its machine-readable result."""
    _load_runtime()
    jax.config.update("jax_enable_x64", True)
    environment = _environment()
    provenance = _provenance(wheel)
    if wheel is not None and provenance["source_tree_import_check"] != "wheel":
        raise RuntimeError(
            f"installed-wheel qualification imported diff_soup from "
            f"{provenance['imported_from']}, not the clean wheel environment"
        )
    cases = [
        _measure(case, args.warmups, args.samples)
        for case in select(args.suite, device=jax.default_backend())
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "suite_version": SUITE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite": args.suite,
        "environment_key": _environment_key(environment),
        "environment": environment,
        "provenance": provenance,
        "cases": cases,
        "status": "passed" if all(case["status"] == "passed" for case in cases) else "failed",
    }
    _validate(result)
    return result


def _run_installed_wheel(args, wheel: Path) -> int:
    """Install a wheel into a clean venv and delegate qualification to it."""
    with tempfile.TemporaryDirectory(prefix="diff-soup-wheel-benchmark-") as directory:
        clean_root = Path(directory)
        environment = clean_root / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = environment / "bin" / "python"
        wheel_device = args.wheel_device
        if wheel_device is None:
            _load_runtime()
            wheel_device = "gpu" if jax.default_backend() == "gpu" else "cpu"
        requirement = f"diff-soup[cu12] @ {wheel.as_uri()}" if wheel_device == "gpu" else str(wheel)
        subprocess.run(
            [str(python), "-m", "pip", "install", requirement, "jsonschema>=4"],
            check=True,
            cwd=clean_root,
        )
        shutil.copytree(
            ROOT / "benchmarks",
            clean_root / "benchmarks",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        child_environment = os.environ.copy()
        commit, dirty = _git_source_state()
        child_environment["DIFF_SOUP_QUALIFICATION_COMMIT"] = commit
        child_environment["DIFF_SOUP_QUALIFICATION_DIRTY"] = "1" if dirty else "0"
        if wheel_device == "cpu":
            child_environment["JAX_PLATFORMS"] = "cpu"
        else:
            child_environment.pop("JAX_PLATFORMS", None)
        command = [
            str(python),
            "-m",
            "benchmarks.run",
            "--suite",
            args.suite,
            "--output",
            str(Path(args.output).resolve()),
            "--warmups",
            str(args.warmups),
            "--samples",
            str(args.samples),
            "--installed-wheel",
            str(wheel.resolve()),
            "--wheel-child",
            "--wheel-device",
            wheel_device,
        ]
        subprocess.run(command, check=True, cwd=clean_root, env=child_environment)
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the benchmark runner command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        required=True,
        choices=("pr-cpu", "nightly-cpu", "nightly-gpu", "release"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--installed-wheel", type=Path)
    parser.add_argument(
        "--wheel-device",
        choices=("cpu", "gpu"),
        help="dependency/runtime target for clean installed-wheel qualification",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--wheel-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Run the requested suite and return nonzero on a hard qualification failure."""
    args = _parser().parse_args()
    if args.warmups < 1 or args.samples < 3:
        raise SystemExit("--warmups must be >= 1 and --samples must be >= 3")
    wheel = args.installed_wheel.resolve() if args.installed_wheel else None
    if wheel and not args.wheel_child:
        return _run_installed_wheel(args, wheel)
    result = _run_suite(args, wheel)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{result['status']}: {args.suite} -> {output}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
