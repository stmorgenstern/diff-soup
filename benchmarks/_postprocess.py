"""Shared validation for benchmark result post-processing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def require_clean_common_commit(results: Sequence[Mapping[str, Any]], *, operation: str) -> None:
    """Reject result sets that cannot share one trustworthy provenance record."""
    if any(result["provenance"]["dirty"] for result in results):
        raise SystemExit(f"{operation} inputs must all come from clean source checkouts")

    recorded = [result["provenance"]["git_commit"] for result in results]
    if any(
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        for commit in recorded
    ):
        raise SystemExit(f"{operation} inputs must record a full source commit")
    commits = set(recorded)
    if len(commits) != 1:
        raise SystemExit(f"{operation} inputs have different source commits")
