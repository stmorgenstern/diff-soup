"""Coordinate patterns: one way to deduplicate the ``(row, col)`` pairs of a Jacobian."""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from typing import NamedTuple, TypeVar

import numpy as np
from jaxtyping import Int


class UniqueCOO(NamedTuple):
    """The distinct entries of a coordinate pattern.

    Attributes:
        rows: Row of each distinct entry.
        cols: Column of each distinct entry.
        inverse: For each input entry, the position of its pair in ``rows`` and ``cols``.
    """

    rows: Int[np.ndarray, " unique"]
    cols: Int[np.ndarray, " unique"]
    inverse: Int[np.ndarray, " entries"]


def dedup_coo(rows, cols, n_cols: int, *, first_seen: bool = False) -> UniqueCOO:
    """Return the distinct ``(row, col)`` pairs of a pattern.

    Args:
        rows: Row indices.
        cols: Column indices, each below ``n_cols``.
        n_cols: The number of columns.
        first_seen: Keep each pair where it first appears instead of in row-major order.

    Returns:
        A [UniqueCOO][diff_soup.problem._ad.pattern.UniqueCOO] with ``int32`` rows and
        columns.
    """
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols = np.asarray(cols, dtype=np.int64).reshape(-1)
    keys = rows * np.int64(n_cols) + cols
    unique, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    if first_seen:
        order = np.argsort(first, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        unique, inverse = unique[order], rank[inverse]
    return UniqueCOO(
        (unique // np.int64(n_cols)).astype(np.int32),
        (unique % np.int64(n_cols)).astype(np.int32),
        inverse.reshape(-1).astype(np.int32),
    )


_T = TypeVar("_T")


def outside_compile_time_eval(fn: Callable[[], _T]) -> _T:
    """Return ``fn()`` run where ``jax.ensure_compile_time_eval`` does not reach.

    JAX's configuration contexts are per thread, so a worker thread runs ``fn`` under
    the process defaults. A residual traced there for its pattern stages its constant
    subexpressions instead of compiling each one, and the solver view builds its
    pattern inside compile-time evaluation.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
        return worker.submit(fn).result()
