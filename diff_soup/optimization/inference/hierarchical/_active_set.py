"""The active set of a step under linearized inequality rows: a primal active-set method.

The rows at the current point are linearized, ``c + C s <= 0``; ``s = 0`` keeps them. The
working set starts with the rows at their boundary, a linearly independent set of them. Each
round solves the step with the working set held at its boundary (``solve``). A step that
crosses a row outside the set moves the iterate along it to the first such row, which joins
the set; a step that crosses none becomes the iterate, and a held row whose multiplier is
negative leaves the set, as the model then gains by moving off it. A row in the span of the
held rows never joins: its linearized value is constant on their boundary, so it is crossed by
rounding only, and holding it too would make the projection singular. The set is settled when
the step crosses no row and no held multiplier is negative: the step is then the solution of
the linearized problem (Nocedal and Wright, Numerical Optimization, 2nd ed., Algorithm 16.3).
On a convex model the iterate never raises the model, so a step taken before the set settles is
still a descent step.

Rows fall into independent groups (one per member in the polish, one for the joint step):
a group's rows read only its own columns among those a step moves, and every group runs the
method on its own, all in lockstep, one ``solve`` per round.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

#: The rounds of the method beyond twice the rows of the largest group (ours): each round adds
#: or releases one row per group, and a set settles within about one round per row.
_EXTRA_ROUNDS = 4
#: A row whose coefficients, normalized, keep less than this norm off the span of the held rows
#: is dependent on them, and never joins the set (ours).
_DEPENDENT = 1e-8


class Active(NamedTuple):
    """Where the method ended.

    Attributes:
        active: ``(R,)``, the working set.
        answer: The last ``solve``'s answer, with the working set ``active``.
        step: ``(n,)``, the iterate: on a settled group's columns the step of ``answer``.
        settled: ``(groups,)``, whether each group's set settled; an unsettled group's
            step is a descent step that is not the solution of its linearized problem.
        solves: The ``solve`` calls made.
    """

    active: np.ndarray
    answer: tuple
    step: np.ndarray
    settled: np.ndarray
    solves: int


def slack(c: np.ndarray, C: np.ndarray, step: np.ndarray) -> np.ndarray:
    """A row's slack for rounding: a few hundred eps of the sizes its linearization adds."""
    eps = float(np.finfo(np.float64).eps)
    return 256.0 * eps * (np.abs(c) + np.abs(C) @ np.abs(step)) + float(np.finfo(np.float64).tiny)


def _independent(C: np.ndarray, held: np.ndarray, row: int) -> bool:
    """Whether row ``row`` of ``C`` keeps a part off the span of the rows ``held``.

    The rows are compared on the columns any of them reads (a member's rows read a few),
    the row normalized.
    """
    columns = np.flatnonzero(np.any(C[np.append(held, row)] != 0.0, axis=0))
    vector = C[row, columns] / np.linalg.norm(C[row, columns])
    if held.size:
        # On the host in float64, whatever JAX's precision: a float32 basis would leave a
        # dependent row a part of its rounding, above the threshold.
        basis, _ = np.linalg.qr(C[np.ix_(held, columns)].T)
        vector = vector - basis @ (basis.T @ vector)
    return float(np.linalg.norm(vector)) > _DEPENDENT


def boundary(
    c: np.ndarray, C: np.ndarray, x: np.ndarray, row_group: np.ndarray, groups: int
) -> np.ndarray:
    """The rows at their boundary at ``x``, independent within each group: the starting set.

    A row is at its boundary within ``sqrt(eps)`` of its size ``|c| + |C| |x|``. A row with
    no column a step moves, or one whose coefficients lie in the span of the group's rows
    taken before it, is left out.
    """
    eps = float(np.finfo(np.float64).eps)
    size = np.abs(c) + np.abs(C) @ np.abs(x)
    near = (c >= -np.sqrt(eps) * size) & np.any(C != 0.0, axis=1)
    chosen = np.zeros(c.shape, dtype=bool)
    for row in np.flatnonzero(near):
        if _independent(C, np.flatnonzero(chosen & (row_group == row_group[row])), int(row)):
            chosen[row] = True
    return chosen


def active_solve(
    solve: Callable,
    multipliers: Callable,
    c: np.ndarray,
    C: np.ndarray,
    x: np.ndarray,
    first: tuple | None = None,
    groups: tuple | None = None,
) -> Active:
    """The step under the rows ``c + C s <= 0`` by the primal active-set method.

    Args:
        solve: ``solve(active)`` gives an answer whose first entry is the step ``(n,)`` in
            joint columns with the rows of ``active`` held at their boundary.
        multipliers: ``multipliers(active, answer)`` gives ``(R,)``, the multipliers of the
            held rows at ``answer`` (zero on the others).
        c: ``(R,)``, the rows at ``x``.
        C: ``(R, n)``, their Jacobian, zero on the columns a step never moves.
        x: ``(n,)``, the current point.
        first: ``solve``'s answer with no row held, when already made.
        groups: ``(row_group (R,), column_group (n,), count)``, the group of every row and
            of every column (``-1`` for a column of no group), or ``None`` for one group.

    A ``solve`` that raises ``ValueError`` (no finite step keeps the set within the radius)
    ends the method with the set and the answer before that round; the groups that changed
    the set in that round are unsettled.

    Returns:
        An [Active][diff_soup.optimization.inference.hierarchical._active_set.Active].
    """
    count = int(c.shape[0])
    if groups is None:
        row_group = np.zeros((count,), dtype=np.int64)
        column_group = np.zeros((C.shape[1],), dtype=np.int64)
        total = 1
    else:
        row_group, column_group, total = groups
    per = np.bincount(row_group, minlength=total)
    limit = _EXTRA_ROUNDS + 2 * int(per.max(initial=0))
    active = boundary(c, C, x, row_group, total)
    solves = 0
    if active.any() or first is None:
        answer, solves = solve(active), 1
    else:
        answer = first
    reached = np.zeros((C.shape[1],), dtype=np.float64)
    settled = np.zeros((total,), dtype=bool)
    for _ in range(limit):
        step = np.asarray(answer[0], dtype=np.float64)
        open_rows = ~settled[row_group]
        with np.errstate(all="ignore"):
            crossed = open_rows & ~active & (c + C @ step > slack(c, C, step))
            move = step - reached
            rate = C @ move
            # The fraction of the move at which a crossed row reaches its boundary.
            fraction = np.maximum(-(c + C @ reached), 0.0) / np.where(rate > 0.0, rate, np.inf)
        fraction = np.where(crossed & np.isfinite(fraction), fraction, 0.0)
        # Per open group, the first row its move reaches, which joins the set. A row in the
        # span of the group's held rows is constant on their boundary, so it is crossed by
        # rounding only and never joins.
        joining = np.full((total,), -1, dtype=np.int64)
        for group in np.flatnonzero(np.bincount(row_group[crossed], minlength=total) > 0):
            held = np.flatnonzero((row_group == group) & active)
            candidates = np.flatnonzero((row_group == group) & crossed)
            for row in candidates[np.argsort(fraction[candidates], kind="stable")]:
                if _independent(C, held, int(row)):
                    joining[group] = int(row)
                    break
        holding = np.bincount(row_group[active], minlength=total) > 0
        mu = np.zeros((count,))
        if (~settled & (joining < 0) & holding).any():
            mu = np.asarray(multipliers(active, answer), dtype=np.float64)
        changed = False
        before = active.copy()
        for group in np.flatnonzero(~settled):
            columns = column_group == group
            row = int(joining[group])
            if row >= 0:
                reached[columns] += min(float(fraction[row]), 1.0) * move[columns]
                active[row] = True
                changed = True
                continue
            reached[columns] = step[columns]
            held = np.flatnonzero((row_group == group) & active)
            values = mu[held]
            if held.size:
                # A multiplier within rounding of the largest one's size counts as zero.
                tolerance = np.sqrt(np.finfo(np.float64).eps) * float(
                    np.nanmax(np.abs(values), initial=0.0)
                )
                if np.nanmin(values, initial=np.inf) < -tolerance:
                    active[int(held[np.nanargmin(values)])] = False
                    changed = True
                    continue
            settled[group] = True
        if not changed:
            break
        try:
            answer = solve(active)
        except ValueError:
            # The groups that changed the set this round are not settled; the others are.
            active = before
            break
        solves += 1
    return Active(active, answer, reached, settled, solves)


__all__ = ["Active", "active_solve", "boundary", "slack"]
