"""Packaging contracts: the benchmark schema and comparator, the docs artifact budget tool."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from benchmarks.compare import compare

_ROOT = Path(__file__).resolve().parents[2]


def _result():
    """The example benchmark result reduced to one passing case."""
    result = json.loads((_ROOT / "benchmarks" / "example_result.json").read_text(encoding="utf-8"))
    for name in ("accepted", "baseline_review", "comparison_policy", "fresh_processes"):
        result.pop(name, None)
    case = copy.deepcopy(result["cases"][0])
    case.update(name="smoke", version="1", status="passed", error=None)
    case["measurements"] = {
        "trace_ms": 1.0,
        "lower_ms": 1.0,
        "compile_ms": 10.0,
        "warmups": 2,
        "raw_ms": [1.0, 1.0, 1.0],
        "median_ms": 1.0,
        "mad_ms": 0.0,
        "p95_ms": 1.0,
        "peak_host_bytes": 1000,
        "peak_device_bytes": 1000,
    }
    case["native_telemetry"] = {"native_serialized_fallback_calls": 0}
    case["correctness"] = {"passed": True}
    case["structure"] = {"passed": True}
    result.update(suite="pr-cpu", cases=[case], status="passed")
    return result


def test_a_benchmark_result_meets_the_schema_and_the_comparator_gates_regressions():
    """A valid result passes the schema; the comparator fails a regressed case, not the baseline."""
    schema = json.loads((_ROOT / "benchmarks" / "schema.json").read_text(encoding="utf-8"))
    baseline = _result()
    jsonschema.validate(baseline, schema)
    uppercase = copy.deepcopy(baseline)
    uppercase["provenance"]["git_commit"] = "A" * 40
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(uppercase, schema)
    baseline["accepted"] = True
    baseline["comparison_policy"] = {
        "warm_runtime_relative": 0.15,
        "compile_time_relative": 0.20,
        "peak_memory_relative": 0.15,
        "peak_device_memory_relative": 0.15,
        "maximum_relative_mad": 0.15,
        "absolute_mad_envelope_ms": {"smoke": 0.05},
        "absolute_noise_floor_ms": {"smoke": 0.01},
        "compile_noise_floor_ms": {"smoke": 0.1},
        "memory_noise_floor_bytes": {"smoke": 10},
        "device_memory_noise_floor_bytes": {"smoke": 10},
        "speedup_rules": [],
    }
    assert compare(baseline, baseline)["status"] == "passed"
    for section, name, value in (
        ("structure", "passed", False),
        ("native_telemetry", "native_serialized_fallback_calls", 1),
        ("measurements", "median_ms", 1.3),
        ("measurements", "peak_host_bytes", 1300),
    ):
        candidate = copy.deepcopy(baseline)
        candidate["cases"][0][section][name] = value
        assert compare(baseline, candidate)["status"] == "failed"


def test_the_docs_artifact_budget_tool_fails_and_names_an_oversize_file(tmp_path):
    """.github/scripts/docs_artifact_budget.py exits 1 and names a file over its hard cap."""
    (tmp_path / "small.png").write_bytes(b"x" * 1000)
    (tmp_path / "huge.png").write_bytes(b"x" * 1_000_001)
    run = subprocess.run(
        [sys.executable, _ROOT / ".github/scripts/docs_artifact_budget.py", "--root", tmp_path],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 1 and "huge.png" in run.stdout and "small.png" not in run.stdout
