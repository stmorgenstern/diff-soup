"""Benchmark case registry."""

from __future__ import annotations

from benchmarks._core import Case
from benchmarks.cases.representative_models import cases as representative_cases
from benchmarks.cases.solver_transforms import cases as solver_cases
from benchmarks.cases.sparse_linear import cases as sparse_cases
from benchmarks.cases.uq_oed import cases as uq_cases


def select(suite: str, *, device: str | None = None) -> tuple[Case, ...]:
    """Return the stable ordered case set for ``suite``."""
    known = {"pr-cpu", "nightly-cpu", "nightly-gpu", "release"}
    if suite not in known:
        raise ValueError(f"unknown benchmark suite {suite!r}; choose one of {sorted(known)}")
    catalog = (
        *solver_cases(),
        *sparse_cases(),
        *uq_cases(),
        *representative_cases(),
    )
    return tuple(
        case
        for case in catalog
        if suite in case.suites and (case.required_device is None or case.required_device == device)
    )


__all__ = ["select"]
