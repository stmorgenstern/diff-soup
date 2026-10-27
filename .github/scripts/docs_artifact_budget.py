"""Enforce web-friendly size budgets on committed documentation artifacts.

Every file under the documentation tree must stay small enough that clones
and page loads stay fast: files over the hard cap fail the build, files over
the warning threshold are listed so growth is visible in CI logs.
"""

import argparse
import sys
from pathlib import Path


def main() -> int:
    """Scan the documentation tree and report size-budget violations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("docs"))
    parser.add_argument("--max-bytes", type=int, default=1_000_000)
    parser.add_argument("--warn-bytes", type=int, default=300_000)
    args = parser.parse_args()

    failures = 0
    for path in sorted(args.root.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > args.max_bytes:
            print(f"FAIL {path} {size} bytes exceeds the {args.max_bytes}-byte cap")
            failures += 1
        elif size > args.warn_bytes:
            print(f"WARN {path} {size} bytes exceeds the {args.warn_bytes}-byte warning threshold")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
