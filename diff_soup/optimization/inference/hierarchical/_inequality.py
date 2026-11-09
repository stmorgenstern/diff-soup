"""Inequality rows on the fitted values of a lowered study, ``c(own, shared, data) <= 0``.

The level trust region keeps them by the smallest extension of its equality rows. At each
step the rows are linearized at the current point, ``c + C s <= 0``; a row the step is
predicted to violate joins the equality rows as ``C_i s = -c_i`` (the boundary), through the
projection every damped solve already makes, and the step is solved again, a few times at
most. The rows ride at a fixed count with a mask of the active ones, so one program solves
every active set. A trial at which a row is violated beyond its tolerance, as the rows are
nonlinear, is scaled back along its step to the boundary, as a trial leaving the box is,
and moved back onto a curved row that its step held on the row's linearization. There is
no barrier, no new solver and no derivative rule.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup.optimization.inference.hierarchical import _level_elimination as elimination
from diff_soup.optimization.inference.hierarchical._level_system import _offsets

#: The largest number of solves of one step's active set: each adds the rows it violates.
_ACTIVE_ROUNDS = 4
#: The bisections of the scale-back of a trial that violates a row.
_STEP_BACK_BISECTIONS = 40
#: The Newton iterations that move a trial back onto the rows it violates.
_RESTORE_ITERATIONS = 3


class InequalityRows(eqx.Module):
    """Rows kept at or below zero on the fitted values of a lowered study, for `TrustRegion(inequalities=...)`.

    With `level`, every member of that level has its own rows `rows(own, shared, data)`:
    `own` is the member's values at the level, `shared` the shared values, both in the
    fitted vector's order, and `data` the member's entry of `data`. Without `level` the rows
    are `rows(shared, data)`. Each call returns the same number of rows.

    Each step holds the rows it would cross at their boundary, and a trial that crosses a
    row is moved back to it, so the fit stays where the rows hold. Start it there.

    Args:
        rows: The rows, a JAX function of the values, each kept at or below zero.
        level: The level whose members' own values the rows read, or `None` for the shared values only.
        data: `{member: array}` for every member of `level` (one array for `level=None`), or `None`.
    """

    rows: Callable = eqx.field(static=True)
    level: Any = eqx.field(static=True)
    data: Any

    def __init__(self, rows: Callable, *, level: Any = None, data: Any = None):
        """Store the rows, the level and the data."""
        if not callable(rows):
            raise ConfigurationError(f"InequalityRows rows must be callable; got {rows!r}.")
        self.rows = rows
        self.level = level
        self.data = data


class RowSet:
    """Every inequality row of a fit, evaluated with its Jacobian in joint columns on the host.

    The rows are ordered spec by spec, and within a spec member by member. ``levels`` holds,
    per level that has rows, ``(positions, columns)``: ``positions[n, r]`` the index of row
    ``r`` of member ``n`` and ``columns[n]`` the member's joint columns at the level.
    """

    def __init__(self, specs: tuple, route: Any):
        """Place every spec's rows on the route's joint columns and compile their evaluation.

        Raises:
            ConfigurationError: If a spec is not an `InequalityRows`, names a level the
                study does not have, or its data miss a member.
        """
        counts, widths, names = route.tables.counts, route.tables.widths, route.names
        shared = int(widths[0])
        self.n = int(route.n)
        self.shared = shared
        self.fixed = np.asarray(route.fixed, dtype=np.int64)
        self.blocks: list = []
        self.levels: dict = {}
        count = 0
        for index, spec in enumerate(specs):
            if not isinstance(spec, InequalityRows):
                raise ConfigurationError(
                    "TrustRegion inequalities= takes InequalityRows instances; "
                    f"entry {index} is {type(spec).__name__}."
                )
            if spec.level is None:
                program = jax.jit(_shared_rows(spec.rows))
                self.blocks.append((program, None, spec.data))
                count += int(np.shape(program(np.zeros((shared,)), spec.data)[0])[0])
                continue
            if spec.level not in names:
                raise ConfigurationError(
                    f"InequalityRows level={spec.level!r} is not a level of this study; its "
                    f"levels are {tuple(names)}."
                )
            level = names.index(spec.level) + 1
            members = tuple(route.members[level - 1])
            data = spec.data
            if data is not None:
                keyed = (
                    {str(key): value for key, value in data.items()}
                    if isinstance(data, Mapping)
                    else {}
                )
                missing = tuple(member for member in members if str(member) not in keyed)
                if missing:
                    raise ConfigurationError(
                        f"InequalityRows data takes {{member: array}} for every member of level "
                        f"{spec.level!r}: {members}; missing {missing}."
                    )
                data = jax.tree.map(
                    lambda *leaves: jnp.stack(leaves), *(keyed[str(member)] for member in members)
                )
            offset = shared + sum(
                int(counts[above - 1]) * int(widths[above]) for above in range(1, level)
            )
            width = int(widths[level])
            columns = offset + np.arange(len(members) * width).reshape(len(members), width)
            axes = (0, None, 0 if data is not None else None)
            program = jax.jit(jax.vmap(_member_rows(spec.rows), in_axes=axes))
            rows = int(np.shape(program(np.zeros(columns.shape), np.zeros((shared,)), data)[0])[1])
            self.blocks.append((program, columns, data))
            positions = count + np.arange(len(members) * rows).reshape(len(members), rows)
            if level in self.levels:
                positions = np.concatenate([self.levels[level][0], positions], axis=1)
            self.levels[level] = (positions, columns)
            count += len(members) * rows
        self.count = count
        self.kept: tuple | None = None

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(c, C)`` at ``x``: the rows ``(R,)`` and their Jacobian ``(R, n)``, fixed columns zero.

        The last point's are kept for the next request at the same point.
        """
        x = np.asarray(x, dtype=np.float64)
        key = x.tobytes()
        if self.kept is None or self.kept[0] != key:
            self.kept = (key, *self._evaluate(x))
        return self.kept[1].copy(), self.kept[2].copy()

    def _evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shared = x[: self.shared]
        values, jacobians = [], []
        for program, columns, data in self.blocks:
            if columns is None:
                c, d_shared = program(shared, data)
                rows = np.zeros((np.shape(c)[0], self.n))
                rows[:, : self.shared] = np.asarray(d_shared)
                values.append(np.asarray(c))
                jacobians.append(rows)
                continue
            c, d_own, d_shared = (np.asarray(part) for part in program(x[columns], shared, data))
            members, count = c.shape
            rows = np.zeros((members, count, self.n))
            rows[:, :, : self.shared] = d_shared
            rows[
                np.arange(members)[:, None, None],
                np.arange(count)[None, :, None],
                columns[:, None, :],
            ] = d_own
            values.append(c.reshape(-1))
            jacobians.append(rows.reshape(members * count, self.n))
        c = np.concatenate(values) if values else np.zeros((0,))
        C = np.concatenate(jacobians) if jacobians else np.zeros((0, self.n))
        if self.fixed.size:
            C[:, self.fixed] = 0.0
        return c, C

    def values(self, x: np.ndarray) -> np.ndarray:
        """The rows at ``x``, ``(R,)``."""
        return self(x)[0]


def _shared_rows(rows: Callable) -> Callable:
    def evaluate(shared, data):
        value = jnp.ravel(jnp.asarray(rows(shared, data)))
        return value, jax.jacfwd(lambda w: jnp.ravel(jnp.asarray(rows(w, data))))(shared)

    return evaluate


def _member_rows(rows: Callable) -> Callable:
    def evaluate(own, shared, data):
        def flat(z, w):
            return jnp.ravel(jnp.asarray(rows(z, w, data)))

        value = flat(own, shared)
        d_own, d_shared = jax.jacfwd(flat, argnums=(0, 1))(own, shared)
        return value, d_own, d_shared

    return evaluate


def tolerance(c: np.ndarray, C: np.ndarray, step: np.ndarray) -> np.ndarray:
    """A row's slack for rounding: a few hundred eps of the sizes its linearization adds."""
    eps = float(np.finfo(np.float64).eps)
    return 256.0 * eps * (np.abs(c) + np.abs(C) @ np.abs(step)) + float(np.finfo(np.float64).tiny)


def frozen(system: elimination.LevelSystem) -> np.ndarray:
    """``(n,)``: the coordinates a step of ``system`` never moves, fixed or of a held node."""
    mask = np.array(system.fixed_mask, dtype=bool)
    offsets = _offsets(system.counts, system.widths)
    for level in range(1, system.depth + 1):
        held = np.asarray(system.held_mask[level], dtype=bool)
        mask[offsets[level] : offsets[level + 1]] |= np.repeat(held, system.widths[level])
    return mask


def with_rows(
    system: elimination.LevelSystem, x: np.ndarray, c: np.ndarray, C: np.ndarray, active: np.ndarray
) -> elimination.LevelSystem:
    """``system`` with every row among its equality rows, the ``active`` ones kept at their boundary.

    Row ``i`` reads ``C_i (x + s) = C_i x - c_i``, the step ``s`` on which its
    linearization at ``x`` is zero, its columns that the step never moves set to zero; a
    row with no column left is never kept. ``system`` itself when no row is kept.
    """
    rows = np.where(frozen(system)[None, :], 0.0, C)
    active = np.asarray(active, dtype=bool) & np.any(rows != 0.0, axis=1)
    if not active.any():
        return system
    dtype = np.dtype(system.root.dtype)
    placed = system.point
    device = placed.sharding if placed.committed else None
    count = int(np.shape(system.constraint_matrix)[0])
    kept = (
        np.ones((count,), dtype=bool)
        if system.constraint_active is None
        else np.asarray(system.constraint_active, dtype=bool)
    )
    matrix = np.concatenate([np.asarray(system.constraint_matrix, dtype=dtype), rows.astype(dtype)])
    rhs = np.concatenate(
        [np.asarray(system.constraint_rhs, dtype=dtype), (rows @ x - c).astype(dtype)]
    )
    return eqx.tree_at(
        lambda s: (s.constraint_matrix, s.constraint_rhs, s.constraint_active),
        system,
        jax.device_put((matrix, rhs, np.concatenate([kept, active])), device),
        is_leaf=lambda leaf: leaf is None,
    )


def active_solve(
    solve: Callable, c: np.ndarray, C: np.ndarray, first: tuple | None = None
) -> tuple:
    """Solve a step under the rows it is predicted to violate, adding them a few rounds at most.

    ``solve(active)`` returns ``(step, *rest)`` with the step in joint columns, solved with the
    rows of ``active`` at their boundary; ``first`` is its answer with no row held, when
    already made. Every row the step's linearization violates beyond its slack joins the set
    and the step is solved again. A round whose solve fails (no finite step keeps the set
    within the radius) ends the rounds with the last answer, which the scale-back then keeps
    inside the rows. Returns ``(active, (step, *rest), rounds)``, ``rounds`` the solves made.
    """
    active = np.zeros(c.shape, dtype=bool)
    rounds = 0
    if first is None:
        first, rounds = solve(active), 1
    answer = first
    for _ in range(_ACTIVE_ROUNDS - 1):
        step = np.asarray(answer[0])
        with np.errstate(all="ignore"):
            violated = ~active & (c + C @ step > tolerance(c, C, step))
        if not violated.any():
            break
        try:
            following = solve(active | violated)
        except ValueError:
            break
        rounds += 1
        if not np.all(np.isfinite(np.asarray(following[0]))):
            break
        active, answer = active | violated, following
    return active, answer, rounds


def _fractions(
    values: Callable, c0: np.ndarray, watched: np.ndarray, groups: np.ndarray, count: int
) -> np.ndarray:
    """Per group, the largest fraction of its step keeping its watched rows within ``max(c0, 0)``.

    ``values(fractions)`` gives the rows at the point moved by every group's fraction of
    its step; ``groups[i]`` is the group of row ``i``, which reads its group's fraction
    alone. A bisection of every group at once, the rows being nonlinear: a group whose step
    crosses a row is scaled back to that row's boundary.
    """
    limit = np.maximum(c0, 0.0) + tolerance(c0, np.zeros((c0.size, 1)), np.zeros((1,)))

    def broken(fractions: np.ndarray) -> np.ndarray:
        crossed = watched & ~(values(fractions) <= limit)
        return np.bincount(groups[crossed], minlength=count) > 0

    high = np.ones((count,))
    low = np.where(broken(high), 0.0, 1.0)
    if np.all(low == 1.0):
        return low
    for _ in range(_STEP_BACK_BISECTIONS):
        middle = np.where(low < 1.0, 0.5 * (low + high), 1.0)
        bad = broken(middle)
        high = np.where(bad, middle, high)
        low = np.where(bad, low, middle)
    return low


def step_back(rows: RowSet, x: np.ndarray, step: np.ndarray, active: np.ndarray) -> float:
    """The largest fraction of ``step`` at which no inactive row is worse than ``max(c(x), 0)``.

    1 when the full step keeps those rows. An active row, held at its boundary by the
    step's linearization, keeps the step's second-order residual, which the next step's
    projection takes back.
    """
    c0 = rows.values(x)
    fraction = _fractions(
        lambda t: rows.values(x + float(t[0]) * step),
        c0,
        ~np.asarray(active, dtype=bool),
        np.zeros(c0.shape, dtype=np.int64),
        1,
    )
    return float(fraction[0])


def onto_rows(rows: RowSet, x: np.ndarray, scale: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """``x`` moved back onto the rows it violates beyond their slack, the coordinates ``moving`` alone.

    A step that holds a curved row on its linearization leaves it by the square of the
    step; the Newton projection ``x - D^-2 C^T (C D^-2 C^T)^-1 c`` onto the violated rows,
    in the metric ``D = diag(scale)`` of the step, takes the trial back to their boundary,
    a few times at most. A row the coordinates ``moving`` do not reach is left as it is.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    for _ in range(_RESTORE_ITERATIONS):
        c, C = rows(x)
        with np.errstate(all="ignore"):
            A = np.where(moving[None, :], C, 0.0) / scale[None, :]
        A = np.where(np.isfinite(A), A, 0.0)
        reached = np.any(A != 0.0, axis=1)
        violated = reached & (c > tolerance(c, np.zeros((c.size, 1)), np.zeros((1,))))
        if not violated.any():
            break
        A = A[violated]
        multipliers = np.asarray(jnp.linalg.solve(A @ A.T, c[violated]))
        with np.errstate(all="ignore"):
            x = x - np.where(moving, (A.T @ multipliers) / scale, 0.0)
    return x


__all__ = ["InequalityRows"]
