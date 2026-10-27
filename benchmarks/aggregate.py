"""Aggregate repeated fresh-process benchmark results without crossing environments."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks._postprocess import require_clean_common_commit
from benchmarks.run import _validate


def main() -> int:
    """Combine synchronized samples and process-level compile measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.inputs) < 3:
        raise SystemExit("aggregation requires at least three fresh-process results")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    for result in results:
        _validate(result)
    require_clean_common_commit(results, operation="aggregation")
    keys = {result["environment_key"] for result in results}
    if len(keys) != 1:
        raise SystemExit(f"cannot aggregate different environment keys: {sorted(keys)}")
    names = [tuple(case["name"] for case in result["cases"]) for result in results]
    if len(set(names)) != 1:
        raise SystemExit("cannot aggregate different case sets or orderings")

    aggregate = copy.deepcopy(results[0])
    aggregate["generated_at"] = datetime.now(timezone.utc).isoformat()
    aggregate["fresh_processes"] = len(results)
    for index, case in enumerate(aggregate["cases"]):
        peers = [result["cases"][index] for result in results]
        if any(peer["status"] != "passed" for peer in peers):
            case["status"] = "failed"
            case["error"] = "; ".join(
                peer["error"] or peer["status"] for peer in peers if peer["status"] != "passed"
            )
            continue
        measurements = [peer["measurements"] for peer in peers]
        assert all(measurement is not None for measurement in measurements)
        raw = [sample for measurement in measurements for sample in measurement["raw_ms"]]
        median = statistics.median(raw)
        case["measurements"] = {
            "trace_ms": statistics.median(measurement["trace_ms"] for measurement in measurements),
            "lower_ms": statistics.median(measurement["lower_ms"] for measurement in measurements),
            "compile_ms": statistics.median(
                measurement["compile_ms"] for measurement in measurements
            ),
            "warmups": sum(measurement["warmups"] for measurement in measurements),
            "raw_ms": raw,
            "median_ms": median,
            "mad_ms": statistics.median(abs(sample - median) for sample in raw),
            "p95_ms": float(np.percentile(np.asarray(raw), 95)),
            "peak_host_bytes": max(measurement["peak_host_bytes"] for measurement in measurements),
            "peak_device_bytes": max(
                (
                    measurement["peak_device_bytes"]
                    for measurement in measurements
                    if measurement["peak_device_bytes"] is not None
                ),
                default=None,
            ),
        }
        case["correctness"]["passed"] = all(
            peer["correctness"].get("passed", False) for peer in peers
        )
        case["structure"]["passed"] = all(peer["structure"].get("passed", False) for peer in peers)
        telemetry_keys = set().union(*(peer["native_telemetry"] for peer in peers))
        case["native_telemetry"] = {
            key: (
                sum(
                    peer["native_telemetry"].get(key, 0)
                    for peer in peers
                    if isinstance(peer["native_telemetry"].get(key, 0), (int, float))
                )
                if all(
                    isinstance(peer["native_telemetry"].get(key, 0), (int, float)) for peer in peers
                )
                else peers[-1]["native_telemetry"].get(key)
            )
            for key in telemetry_keys
        }
    aggregate["status"] = (
        "passed" if all(case["status"] == "passed" for case in aggregate["cases"]) else "failed"
    )
    _validate(aggregate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{aggregate['status']}: aggregated {len(results)} fresh processes -> {args.output}")
    return 0 if aggregate["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
