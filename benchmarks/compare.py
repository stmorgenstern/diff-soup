"""Compare a candidate benchmark result with an immutable accepted baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "benchmarks" / "schema.json"


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("benchmark comparison requires jsonschema>=4") from exc
    jsonschema.validate(data, json.loads(SCHEMA.read_text(encoding="utf-8")))
    return data


def _case_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["name"]: case for case in result["cases"]}


def _serialized_calls(case: dict[str, Any]) -> int:
    telemetry = case["native_telemetry"]
    return int(
        telemetry.get(
            "native_serialized_fallback_calls",
            telemetry.get("serialized_batch_calls", 0),
        )
    )


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Classify correctness, structure, timing, compile, memory, and variance."""
    hard: list[str] = []
    regressions: list[str] = []
    inconclusive: list[str] = []
    improvements: list[str] = []

    if not baseline.get("accepted", False):
        hard.append("baseline is not marked accepted")
    if baseline["environment_key"] != candidate["environment_key"]:
        hard.append(
            "environment key mismatch: "
            f"{baseline['environment_key']} != {candidate['environment_key']}"
        )
        return {
            "status": "failed",
            "hard_failures": hard,
            "regressions": regressions,
            "inconclusive": inconclusive,
            "improvements": improvements,
            "cases": [],
        }

    policy = baseline.get("comparison_policy", {})
    warm_relative = float(policy.get("warm_runtime_relative", 0.15))
    compile_relative = float(policy.get("compile_time_relative", 0.20))
    memory_relative = float(policy.get("peak_memory_relative", 0.15))
    device_memory_relative = float(policy.get("peak_device_memory_relative", 0.15))
    max_mad = float(policy.get("maximum_relative_mad", 0.15))
    mad_envelopes = policy.get("absolute_mad_envelope_ms", {})
    warm_floors = policy.get("absolute_noise_floor_ms", {})
    compile_floors = policy.get("compile_noise_floor_ms", {})
    memory_floors = policy.get("memory_noise_floor_bytes", {})
    device_memory_floors = policy.get("device_memory_noise_floor_bytes", {})
    base_cases = _case_map(baseline)
    new_cases = _case_map(candidate)
    if set(base_cases) != set(new_cases):
        hard.append(
            f"case set changed: missing={sorted(set(base_cases) - set(new_cases))}, "
            f"unexpected={sorted(set(new_cases) - set(base_cases))}"
        )

    rows = []
    for name in sorted(set(base_cases) & set(new_cases)):
        base = base_cases[name]
        new = new_cases[name]
        case_failures = []
        if new["status"] != "passed":
            case_failures.append(f"status={new['status']}: {new.get('error')}")
        if not new["correctness"].get("passed", False):
            case_failures.append("correctness failed")
        if not new["structure"].get("passed", False):
            case_failures.append("structural lowering failed")
        if _serialized_calls(new) > 0:
            case_failures.append(f"serialized native batch calls={_serialized_calls(new)}")
        if case_failures:
            hard.extend(f"{name}: {failure}" for failure in case_failures)
            continue
        if base["version"] != new["version"]:
            hard.append(f"{name}: case version changed {base['version']} -> {new['version']}")
            continue

        base_measurement = base["measurements"]
        new_measurement = new["measurements"]
        if base_measurement is None or new_measurement is None:
            hard.append(f"{name}: missing measurements")
            continue
        relative_mad = new_measurement["mad_ms"] / max(new_measurement["median_ms"], 1e-12)
        if relative_mad > max_mad:
            absolute_envelope = mad_envelopes.get(name)
            if absolute_envelope is None:
                inconclusive.append(f"{name}: no calibrated absolute MAD envelope")
            elif new_measurement["mad_ms"] > float(absolute_envelope):
                inconclusive.append(
                    f"{name}: relative MAD {relative_mad:.3f} exceeds {max_mad:.3f} "
                    f"and absolute MAD {new_measurement['mad_ms']:.6g} ms exceeds "
                    f"{float(absolute_envelope):.6g} ms"
                )

        classifications = {}
        comparisons = (
            (
                "warm",
                "median_ms",
                warm_relative,
                warm_floors.get(name),
            ),
            (
                "compile",
                "compile_ms",
                compile_relative,
                compile_floors.get(name),
            ),
            (
                "host_memory",
                "peak_host_bytes",
                memory_relative,
                memory_floors.get(name),
            ),
        )
        if (
            base_measurement["peak_device_bytes"] is not None
            or new_measurement["peak_device_bytes"] is not None
            or name in device_memory_floors
        ):
            comparisons += (
                (
                    "device_memory",
                    "peak_device_bytes",
                    device_memory_relative,
                    device_memory_floors.get(name),
                ),
            )
        for label, field, relative_limit, floor in comparisons:
            if floor is None:
                inconclusive.append(f"{name}: no calibrated {label} noise floor")
                classifications[label] = "inconclusive"
                continue
            before_value = base_measurement[field]
            after_value = new_measurement[field]
            if before_value is None or after_value is None:
                hard.append(f"{name}: missing {label} measurement")
                classifications[label] = "failed"
                continue
            before = float(before_value)
            after = float(after_value)
            delta = after - before
            if delta > before * relative_limit and delta > float(floor):
                regressions.append(
                    f"{name}: {label} {before:.6g} -> {after:.6g} "
                    f"(+{delta / max(before, 1e-12):.1%})"
                )
                classifications[label] = "regression"
            elif after < before and before - after > float(floor):
                improvements.append(f"{name}: {label} {before:.6g} -> {after:.6g}")
                classifications[label] = "improvement"
            else:
                classifications[label] = "stable"
        rows.append({"name": name, **classifications})

    for rule in policy.get("speedup_rules", []):
        faster = new_cases.get(rule["faster"])
        slower = new_cases.get(rule["slower"])
        if not faster or not slower or not faster["measurements"] or not slower["measurements"]:
            hard.append(f"speedup rule has missing case: {rule}")
            continue
        speedup = slower["measurements"]["median_ms"] / faster["measurements"]["median_ms"]
        if speedup < float(rule["minimum"]):
            regressions.append(
                f"{rule['faster']} speedup over {rule['slower']} is {speedup:.2f}x; "
                f"require {float(rule['minimum']):.2f}x"
            )

    status = "failed" if hard or regressions else "inconclusive" if inconclusive else "passed"
    return {
        "status": status,
        "hard_failures": hard,
        "regressions": regressions,
        "inconclusive": inconclusive,
        "improvements": improvements,
        "cases": rows,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [f"# Performance comparison: {report['status'].upper()}", ""]
    for title, key in (
        ("Hard failures", "hard_failures"),
        ("Regressions", "regressions"),
        ("Inconclusive", "inconclusive"),
        ("Improvements", "improvements"),
    ):
        lines.extend((f"## {title}", ""))
        values = report[key]
        lines.extend(f"- {value}" for value in values)
        if not values:
            lines.append("- None")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Compare two schema-valid results and enforce the regression policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="an accepted baseline result; accepted baselines are archived outside "
        "this repository, so pass the path you fetched it to",
    )
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--markdown-report", type=Path)
    args = parser.parse_args()
    report = compare(_load(args.baseline), _load(args.candidate))
    rendered = _markdown(report)
    print(rendered)
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_report:
        args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
