"""Calibrate absolute noise floors from repeated same-environment processes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from benchmarks._postprocess import require_clean_common_commit
from benchmarks.run import _validate


def _floor(values) -> float:
    median = statistics.median(values)
    return 3.0 * statistics.median(abs(value - median) for value in values)


def _mad(values) -> float:
    median = statistics.median(values)
    return statistics.median(abs(value - median) for value in values)


def main() -> int:
    """Write a reviewable policy; never mutate or promote a baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.inputs) < 3:
        raise SystemExit("calibration requires at least three fresh-process results")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    for result in results:
        _validate(result)
        if result["status"] != "passed":
            raise SystemExit("calibration inputs must all pass")
    require_clean_common_commit(results, operation="calibration")
    if len({result["environment_key"] for result in results}) != 1:
        raise SystemExit("calibration inputs have different environment keys")
    names = [tuple(case["name"] for case in result["cases"]) for result in results]
    if len(set(names)) != 1:
        raise SystemExit("calibration inputs have different case sets")

    policy = json.loads(args.template.read_text(encoding="utf-8"))
    policy["absolute_mad_envelope_ms"] = {}
    policy["absolute_noise_floor_ms"] = {}
    policy["compile_noise_floor_ms"] = {}
    policy["memory_noise_floor_bytes"] = {}
    policy["device_memory_noise_floor_bytes"] = {}
    for index, name in enumerate(names[0]):
        measurements = [result["cases"][index]["measurements"] for result in results]
        combined_samples = [
            sample for measurement in measurements for sample in measurement["raw_ms"]
        ]
        policy["absolute_mad_envelope_ms"][name] = max(
            _mad(combined_samples), *(measurement["mad_ms"] for measurement in measurements)
        )
        policy["absolute_noise_floor_ms"][name] = _floor(
            [measurement["median_ms"] for measurement in measurements]
        )
        policy["compile_noise_floor_ms"][name] = _floor(
            [measurement["compile_ms"] for measurement in measurements]
        )
        policy["memory_noise_floor_bytes"][name] = int(
            _floor([measurement["peak_host_bytes"] for measurement in measurements])
        )
        device_peaks = [measurement["peak_device_bytes"] for measurement in measurements]
        if any(peak is None for peak in device_peaks) and not all(
            peak is None for peak in device_peaks
        ):
            raise SystemExit(f"calibration inputs inconsistently report device memory for {name}")
        if all(peak is not None for peak in device_peaks):
            policy["device_memory_noise_floor_bytes"][name] = int(_floor(device_peaks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"calibrated {len(names[0])} case-specific noise floors -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
