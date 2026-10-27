"""The status one lane reports with its result.

A lane is one experiment solved at one parameter point. Its solve either
reaches every sample the lane asked for, or it fails, and a failure is part of
the result: it names what went wrong and how far the solve got. No failure is
encoded in a residual, a cost or a derivative; those are zero for a failed
lane, and a reduction over lanes refuses a failed lane unless its caller names
what to do with it.

A unit that a request leaves out (``Request(units=...)``) is not solved at all:
it reports the kind ``absent``, which is neither a success nor a failure, and
it lies outside every sum.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._solution import RESULT

Array = jax.Array

#: What a lane can report, by code. ``ok`` is the only success; ``absent`` marks a unit that
#: was not evaluated, which is neither a success nor a failure.
KINDS: tuple[str, ...] = (
    "ok",
    "max_steps",
    "step_collapse",
    "nonlinear_diverged",
    "linear_fail",
    "singular",
    "inconsistent_ic",
    "accuracy_limit",
    "unknown",
    "sample_not_reached",
    "not_finite",
    "absent",
)

#: The code of ``absent`` in ``KINDS``: a unit the request left out, with no result.
ABSENT: int = KINDS.index("absent")

_SOLVER_KINDS: tuple[tuple[str, str], ...] = (
    ("max_steps", "MAX_STEPS"),
    ("step_collapse", "STEP_COLLAPSE"),
    ("nonlinear_diverged", "NONLINEAR_DIVERGED"),
    ("linear_fail", "LINEAR_FAIL"),
    ("singular", "SINGULAR"),
    ("inconsistent_ic", "INCONSISTENT_IC"),
    ("accuracy_limit", "ACCURACY_LIMIT"),
    ("unknown", "UNKNOWN"),
)


class LaneStatus(eqx.Module):
    """How the solve of one lane ended.

    Every field is an array, so a batch of lanes carries one status per lane.

    Attributes:
        kind: Index into [KINDS][diff_soup.inference.engine._status.KINDS];
            zero means the lane solved.
        t_reached: The time the solve reached.
        rung: The rung that produced this result: 0 for the solve from the
            lane's own first step, ``r`` for the solve from
            ``retry_first_step_factors[r - 1]`` times that first step, after
            a failure of a kind of ``retry_kinds``. A lane kernel reports 0;
            the evaluation of a request sets the rung of each result.
    """

    kind: Array
    t_reached: Array
    rung: Array

    @property
    def ok(self) -> Array:
        """Whether the lane solved."""
        return self.kind == 0

    @property
    def failed(self) -> Array:
        """Whether the lane failed; a unit that was not evaluated (``absent``) did not."""
        return (self.kind != 0) & (self.kind != ABSENT)

    @property
    def absent(self) -> Array:
        """Whether the unit was left out of its request and has no result."""
        return self.kind == ABSENT


def solver_kind(solution: Any) -> Array:
    """The kind code of a solution's terminal status; zero when it succeeded."""
    result = solution.result
    kind = jnp.where(solution.ok, 0, KINDS.index("unknown"))
    for name, member in _SOLVER_KINDS:
        item = getattr(RESULT, member, None)
        if item is not None:
            kind = jnp.where(result == item, KINDS.index(name), kind)
    return jnp.asarray(kind, dtype=jnp.int32)


def lane_status(solution: Any, *, sample_missing: Array, not_finite: Array) -> LaneStatus:
    """The status of a lane from its solution and the checks on its samples.

    A solver failure takes precedence, then a requested sample the solve never
    reached, then a sample or residual row that is not finite.

    Args:
        solution: The solution of the lane's solve.
        sample_missing: Whether a sample the lane asked for is invalid.
        not_finite: Whether a requested sample or a residual row is not finite.
    """
    kind = solver_kind(solution)
    solved = kind == 0
    kind = jnp.where(solved & jnp.asarray(not_finite, dtype=bool), KINDS.index("not_finite"), kind)
    kind = jnp.where(
        solved & jnp.asarray(sample_missing, dtype=bool), KINDS.index("sample_not_reached"), kind
    )
    stats = getattr(solution, "stats", None)
    reached = getattr(stats, "t_reached", None)
    reached = jnp.asarray(jnp.nan) if reached is None else jnp.asarray(reached)
    return LaneStatus(
        kind=jnp.asarray(kind, dtype=jnp.int32),
        t_reached=jax.lax.stop_gradient(reached),
        rung=jnp.zeros((), dtype=jnp.int32),
    )


def failures(status: LaneStatus, lane_ids: tuple) -> tuple[dict, ...]:
    """The failed lanes of a batch, as host records in lane order.

    Args:
        status: The status of a batch of lanes, one entry per lane.
        lane_ids: The identifier of each lane, in the order of the batch.

    Returns:
        One record per failed lane: ``lane``, ``kind``, ``t_reached``, ``rung``.
        A unit that was not evaluated (``absent``) is not a failure and has no
        record.
    """
    kinds = np.asarray(status.kind).reshape(-1)
    reached = np.asarray(status.t_reached).reshape(-1)
    rungs = np.asarray(status.rung).reshape(-1)
    if len(lane_ids) != kinds.shape[0]:
        raise ValueError(
            f"{len(lane_ids)} lane identifiers were given for a status of {kinds.shape[0]} "
            "lanes; pass one identifier per lane."
        )
    return tuple(
        {
            "lane": lane_ids[index],
            "kind": KINDS[int(kinds[index])],
            "t_reached": float(reached[index]),
            "rung": int(rungs[index]),
        }
        for index in range(kinds.shape[0])
        if int(kinds[index]) not in (0, ABSENT)
    )


__all__ = ["ABSENT", "KINDS", "LaneStatus", "failures", "lane_status", "solver_kind"]
