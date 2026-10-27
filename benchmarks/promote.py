"""Promote a reviewed, passing candidate into an immutable benchmark baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.run import _validate


def main() -> int:
    """Validate explicit governance inputs and write an accepted baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--comparison-report", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="where to write the accepted baseline; accepted baselines are kept "
        "outside this repository, not under benchmarks/",
    )
    parser.add_argument("--reason", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--environment-key", required=True)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = json.loads(args.comparison_report.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    _validate(candidate)
    if candidate["status"] != "passed":
        raise SystemExit("refusing to promote a candidate whose suite did not pass")
    if candidate["environment_key"] != args.environment_key:
        raise SystemExit("candidate environment key does not match --environment-key")
    if report.get("status") != "passed":
        raise SystemExit("refusing to promote without a passing before/after comparison")
    if candidate["provenance"]["dirty"]:
        raise SystemExit("refusing to promote a dirty source checkout")
    if args.output.exists():
        raise SystemExit("baseline output already exists; preserve history and choose a new path")

    candidate["accepted"] = True
    candidate["comparison_policy"] = policy
    candidate["baseline_review"] = {
        "reason": args.reason,
        "reviewer": args.reviewer,
        "comparison_report": str(args.comparison_report),
    }
    # Baselines are written once. Review and rename the new file in a normal
    # source-control change; benchmark jobs never call this command.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"promoted accepted baseline: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
