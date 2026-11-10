"""Inequality rows on the fitted values of a lowered study, ``c(own, shared, data) <= 0``.

The level trust region keeps them by the smallest extension of its equality rows. At each
step the rows are linearized at the current point, ``c + C s <= 0``, and the step's active
set is found by the primal active-set method (``_active_set``): a held row joins the
equality rows as ``C_i s = -c_i`` (the boundary), through the projection every damped solve
already makes, a row the step would cross joins the set and a held row whose multiplier is
negative leaves it. The rows ride at a fixed count with a mask of the active ones, so one
program solves every active set. A trial at which a row is violated beyond its tolerance, as
the rows are nonlinear, is scaled back along its step to the boundary, as a trial leaving the
box is, and moved back onto a curved row that its step held on the row's linearization. The
joint step keeps every row; a member's polish step keeps the member's own rows at the cut
level, the values above it held. There is no barrier, no new solver and no derivative rule.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup.optimization.inference.hierarchical import _bounds as box
from diff_soup.optimization.inference.hierarchical import _level_elimination as elimination
from diff_soup.optimization.inference.hierarchical._active_set import active_solve, slack
from diff_soup.optimization.inference.hierarchical._level_system import _offsets

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

    Each step holds at their boundary the rows it would cross, and releases a held row the
    fit gains by leaving; a trial that crosses a row is moved back to it, so the fit stays
    where the rows hold. Start it there.

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

    def __init__(self, specs: tuple, route: Any, cut: int | None = None):
        """Place every spec's rows on the route's joint columns and compile their evaluation.

        ``cut`` is the nested mode's cut level (``1 .. D``), or ``None``.

        Raises:
            ConfigurationError: If a spec is not an `InequalityRows`, names a level the
                study does not have or one below the cut, or its data miss a member.
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
            if cut is not None and level > cut:
                raise ConfigurationError(
                    f"InequalityRows level={spec.level!r} lies below the nested mode's cut "
                    f"{names[cut - 1]!r}, and the members' polish keeps rows on the values at "
                    "the cut or above only; put the rows at that level, or fit with "
                    "nested_at=None."
                )
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


def frozen(system: elimination.LevelSystem) -> np.ndarray:
    """``(n,)``: the coordinates a step of ``system`` never moves, fixed or of a held node."""
    mask = np.array(system.fixed_mask, dtype=bool)
    offsets = _offsets(system.counts, system.widths)
    for level in range(1, system.depth + 1):
        held = np.asarray(system.held_mask[level], dtype=bool)
        mask[offsets[level] : offsets[level + 1]] |= np.repeat(held, system.widths[level])
    return mask


def _general_rows(rows: RowSet, system: elimination.LevelSystem) -> np.ndarray:
    """The rows of no member, in order: those read the shared values only, or a level of no width."""
    general = np.ones((rows.count,), dtype=bool)
    for level, (positions, _) in rows.levels.items():
        if system.widths[level]:
            general[positions.reshape(-1)] = False
    return np.flatnonzero(general)


def _member_levels(rows: RowSet, system: elimination.LevelSystem) -> tuple:
    """The levels whose rows are kept at their members' nodes, in increasing order."""
    return tuple(level for level in sorted(rows.levels) if system.widths[level])


def _place(
    rows: RowSet,
    system: elimination.LevelSystem,
    c: np.ndarray,
    C: np.ndarray,
    active: np.ndarray,
    level: int,
) -> elimination.MemberRows:
    """The members' rows of ``level`` as ``MemberRows``: their coefficients from ``C`` per node, target ``-c``.

    Row ``i`` holds ``C_i s = -c_i``, the step on which its linearization at the point of
    ``c`` and ``C`` is zero; ``C`` is read on each member's own columns and on those of its
    ancestors. Host arrays of the system's dtype.
    """
    dtype = np.dtype(system.root.dtype)
    positions, columns = rows.levels[level]
    count, per = positions.shape
    offsets = _offsets(system.counts, system.widths)
    block = C[positions]
    nodes = np.arange(count)[:, None, None]
    numbers = np.arange(per)[None, :, None]

    def read(read_columns: np.ndarray) -> np.ndarray:
        return np.transpose(block[nodes, numbers, read_columns[:, None, :]], (0, 2, 1)).astype(
            dtype
        )

    above = []
    for at in range(level):
        width = system.widths[at]
        if at == 0:
            node_columns = np.broadcast_to(np.arange(width), (count, width))
        else:
            ancestor = np.asarray(system.ancestors[level][at], dtype=np.int64)
            node_columns = offsets[at] + ancestor[:, None] * width + np.arange(width)[None, :]
        above.append(read(node_columns))
    return elimination.MemberRows(
        coefficients=read(columns),
        above=tuple(above),
        target=(-c[positions]).astype(dtype),
        active=np.asarray(active, dtype=bool)[positions],
    )


def _holding(
    system: elimination.LevelSystem, member_rows: tuple, **fields
) -> elimination.LevelSystem:
    """``system`` with ``member_rows`` (one entry per level) and any other ``fields``, on its device."""
    placed = system.point
    device = placed.sharding if placed.committed else None
    names = ("member_rows",) + tuple(fields)
    values = (member_rows,) + tuple(fields.values())
    return eqx.tree_at(
        lambda s: tuple(getattr(s, name) for name in names),
        system,
        jax.device_put(values, device),
        is_leaf=lambda leaf: leaf is None,
    )


def with_rows(
    rows: RowSet,
    system: elimination.LevelSystem,
    x: np.ndarray,
    c: np.ndarray,
    C: np.ndarray,
    active: np.ndarray,
) -> elimination.LevelSystem:
    """``system`` holding every row of ``rows``, the ``active`` ones kept at their boundary.

    Row ``i`` reads ``C_i (x + s) = C_i x - c_i``, the step ``s`` on which its
    linearization at ``x`` is zero, its columns that the step never moves set to zero; a
    row with no column left is never kept. A member's rows ride at its node
    (``member_rows``), the others among the equality rows. ``system`` itself when no row
    is kept.
    """
    C = np.where(frozen(system)[None, :], 0.0, C)
    active = np.asarray(active, dtype=bool) & np.any(C != 0.0, axis=1)
    if not active.any():
        return system
    dtype = np.dtype(system.root.dtype)
    member_rows: list = [None] * (system.depth + 1)
    for level in _member_levels(rows, system):
        member_rows[level] = _place(rows, system, c, C, active, level)
    general = _general_rows(rows, system)
    if not general.size:
        return _holding(system, tuple(member_rows))
    count = int(np.shape(system.constraint_matrix)[0])
    kept = (
        np.ones((count,), dtype=bool)
        if system.constraint_active is None
        else np.asarray(system.constraint_active, dtype=bool)
    )
    matrix = np.concatenate(
        [np.asarray(system.constraint_matrix, dtype=dtype), C[general].astype(dtype)]
    )
    rhs = np.concatenate(
        [
            np.asarray(system.constraint_rhs, dtype=dtype),
            (C[general] @ x - c[general]).astype(dtype),
        ]
    )
    return _holding(
        system,
        tuple(member_rows) if any(entry is not None for entry in member_rows) else None,
        constraint_matrix=matrix,
        constraint_rhs=rhs,
        constraint_active=np.concatenate([kept, active[general]]),
    )


def held_multipliers(
    rows: RowSet,
    system: elimination.LevelSystem,
    held: elimination.LevelSystem,
    coordinates: box.Scaling,
    damping: float,
) -> np.ndarray:
    """``(R,)``: the multipliers of the rows ``held`` keeps (``with_rows``), zero on the others.

    They are those of ``held``'s damped solve in the subproblem's ``coordinates``, in the
    order of ``rows``; ``held`` is ``system`` itself when it keeps no row.
    """
    if held is system:
        return np.zeros((rows.count,))
    flat = elimination.row_multipliers(held, coordinates.scale, damping, coordinates.addition)
    multipliers = np.zeros((rows.count,))
    start, stop = int(system.constraint_matrix.shape[0]), int(held.constraint_matrix.shape[0])
    multipliers[_general_rows(rows, system)] = flat[start:stop]
    for level in _member_levels(rows, system):
        positions = rows.levels[level][0].reshape(-1)
        multipliers[positions] = flat[stop : stop + positions.size]
        stop += positions.size
    return multipliers


def _fractions(
    values: Callable, c0: np.ndarray, watched: np.ndarray, groups: np.ndarray, count: int
) -> np.ndarray:
    """Per group, the largest fraction of its step keeping its watched rows within ``max(c0, 0)``.

    ``values(fractions)`` gives the rows at the point moved by every group's fraction of
    its step; ``groups[i]`` is the group of row ``i``, which reads its group's fraction
    alone. A bisection of every group at once, the rows being nonlinear: a group whose step
    crosses a row is scaled back to that row's boundary.
    """
    limit = np.maximum(c0, 0.0) + slack(c0, np.zeros((c0.size, 1)), np.zeros((1,)))

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


def onto_rows(
    rows: RowSet,
    x: np.ndarray,
    scale: np.ndarray,
    moving: np.ndarray,
    bounds: tuple | None = None,
) -> np.ndarray:
    """``x`` moved back onto the rows it violates beyond their slack, the coordinates ``moving`` alone.

    A step that holds a curved row on its linearization leaves it by the square of the
    step; the Newton projection ``x - D^-2 C^T (C D^-2 C^T)^-1 c`` onto the violated rows,
    in the metric ``D = diag(scale)`` of the step, takes the trial back to their boundary,
    a few times at most. A row the coordinates ``moving`` do not reach is left as it is.
    Under ``bounds`` a coordinate the projection takes out of the box is put back strictly
    inside it and held, and the projection goes on with the others.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    moving = np.array(moving, dtype=bool, copy=True)
    left = _RESTORE_ITERATIONS
    while left:
        left -= 1
        c, C = rows(x)
        with np.errstate(all="ignore"):
            A = np.where(moving[None, :], C, 0.0) / scale[None, :]
        A = np.where(np.isfinite(A), A, 0.0)
        reached = np.any(A != 0.0, axis=1)
        violated = reached & (c > slack(c, np.zeros((c.size, 1)), np.zeros((1,))))
        if not violated.any():
            break
        A = A[violated]
        multipliers = np.asarray(jnp.linalg.solve(A @ A.T, c[violated]))
        with np.errstate(all="ignore"):
            x = x - np.where(moving, (A.T @ multipliers) / scale, 0.0)
        if bounds is None:
            continue
        lower, upper = bounds
        out = moving & ~((x > lower) & (x < upper))
        if out.any():
            x = np.where(out, box.make_strictly_feasible(x, lower, upper, rstep=0), x)
            moving &= ~out
            left = _RESTORE_ITERATIONS
    return x


# ---------------------------------------------------------------------------
# A member's own rows in the polish
# ---------------------------------------------------------------------------


def member_rows(
    rows: RowSet, system: elimination.LevelSystem, x: np.ndarray, level: int
) -> tuple | None:
    """The rows of the members of ``level`` on their own values at ``x``, the values above held.

    Returns ``(c, C, placed)``: the rows ``(N R,)`` and their Jacobian ``(N R, n)`` member
    by member, with the columns a step never moves set to zero, and the rows as the
    system holds them (``_place``, every row active); ``None`` when the level has no rows.
    """
    if level not in rows.levels or not system.widths[level]:
        return None
    positions = rows.levels[level][0]
    c, C = rows(x)
    C = np.where(frozen(system)[None, :], 0.0, C)
    placed = _place(rows, system, c, C, np.ones(c.shape, dtype=bool), level)
    count, per = positions.shape
    return c[positions].reshape(-1), C[positions].reshape(count * per, -1), placed


def _held(own: tuple, system: elimination.LevelSystem, level: int, active: np.ndarray):
    """``system`` holding the members' rows of ``own`` with ``active`` kept, or ``None`` when none is.

    A member's polish keeps a row only where it reads the member's own values.
    """
    _, _, placed = own
    count, _, per = placed.coefficients.shape
    on = np.asarray(active).reshape(count, per) & np.any(placed.coefficients != 0.0, axis=1)
    if not on.any():
        return None
    member_rows = [None] * (system.depth + 1)
    member_rows[level] = placed._replace(active=on)
    return _holding(system, tuple(member_rows))


def _groups(own: tuple, system: elimination.LevelSystem, level: int) -> tuple:
    """``active_solve``'s rows and groups for the members' polish: one group per member.

    Returns ``(c, C, groups)``, ``C`` zero on the columns above ``level``, which the polish
    holds.
    """
    c, C, placed = own
    count, _, per = placed.coefficients.shape
    owners = elimination.node_of_columns(system, level)
    return (
        c,
        np.where((owners >= 0)[None, :], C, 0.0),
        (np.repeat(np.arange(count), per), owners, count),
    )


def _member_multipliers(
    own: tuple, system, coordinates: box.Scaling, level: int, active: np.ndarray, damping
) -> np.ndarray:
    """Every member's multipliers of its held rows at its damped solve, zero on the others."""
    held = _held(own, system, level, active)
    if held is None:
        return np.zeros(own[0].shape)
    multipliers = elimination.inner_row_multipliers(
        held,
        coordinates.scale,
        np.nan_to_num(np.asarray(damping, dtype=np.float64)),
        level,
        addition=coordinates.addition,
    )
    return multipliers.reshape(-1)


def _open_columns(system, level: int, settled: np.ndarray) -> np.ndarray:
    """``(n,)``: the columns of the members whose active set did not settle."""
    owners = elimination.node_of_columns(system, level)
    return (owners >= 0) & ~settled[np.maximum(owners, 0)]


def member_zero(own: tuple, system, x: np.ndarray, coordinates: box.Scaling, level: int) -> tuple:
    """Every member's solve at zero damping under its own rows, by ``active_solve``.

    Returns ``(zero, first, solves)``: the solve ``(z, q_squared)``, the solve with no row
    held (the first solve of the members' searches), and the solves made. A member whose
    active set did not settle has ``q_squared`` ``nan``: its model's decrease is not known.
    """
    c, C, groups = _groups(own, system, level)
    scale, addition = coordinates.scale, coordinates.addition
    count = own[2].coefficients.shape[0]
    made = [0]

    def solve(active):
        held = _held(own, system, level, active)
        zero = elimination.inner_solve_scaled(
            system if held is None else held, scale, level, addition=addition
        )
        with np.errstate(all="ignore"):
            return zero[0] / scale, zero

    def multipliers(active, answer):
        made[0] += 1
        return _member_multipliers(own, system, coordinates, level, active, np.zeros((count,)))

    first = solve(np.zeros(c.shape, dtype=bool))
    found = active_solve(solve, multipliers, c, C, x, first, groups)
    z, q_squared = (np.array(part, dtype=np.float64) for part in found.answer[1])
    if not found.settled.all():
        z = np.where(_open_columns(system, level, found.settled), scale * found.step, z)
        q_squared = np.where(found.settled, q_squared, np.nan)
    return (z, q_squared), first[1], found.solves + 1 + made[0]


def member_predicted(system, step: np.ndarray, coordinates: box.Scaling, level: int) -> np.ndarray:
    """The decrease every member's scaled model predicts for its part of ``step``."""
    predicted = elimination.inner_predicted(system, step, level)
    if coordinates.addition is not None:
        predicted -= 0.5 * elimination.inner_added(
            system, coordinates.addition, coordinates.scale * step, level
        )
    return predicted


def member_step(
    own: tuple,
    system,
    x: np.ndarray,
    gradient: np.ndarray,
    coordinates: box.Scaling,
    radii: np.ndarray,
    level: int,
    members: np.ndarray,
    first: tuple,
    bounds: tuple | None,
) -> tuple:
    """``inner_secular_step`` under every member's own rows by ``active_solve``, and the rows held.

    A member whose active set does not settle takes the method's last iterate, a descent
    step inside its radius that keeps its rows. A member that holds a row and leaves the box
    takes its step shortened to stay inside, which keeps the rows, as the joint step under
    equality rows does. Returns ``(step, predicted, multipliers, solves, active)``.
    """
    c, C, groups = _groups(own, system, level)
    scale, addition = coordinates.scale, coordinates.addition
    made = [0]

    def solve(active):
        held = _held(own, system, level, active)
        answer = elimination.inner_secular_step(
            system if held is None else held,
            scale,
            radii,
            level,
            members,
            first=first if held is None else None,
            addition=addition,
            hold=True,
        )
        made[0] += answer[3]
        return answer

    def multipliers(active, answer):
        made[0] += 1
        return _member_multipliers(own, system, coordinates, level, active, answer[2])

    found = active_solve(solve, multipliers, c, C, x, None, groups)
    active = found.active
    step, predicted, multipliers, _ = found.answer
    if not found.settled.all():
        step = np.where(_open_columns(system, level, found.settled), found.step, step)
        predicted = np.where(
            found.settled, predicted, member_predicted(system, step, coordinates, level)
        )
        predicted[~members | np.isnan(multipliers)] = 0.0
    holding = np.any(active.reshape(len(radii), -1), axis=1) & members
    if bounds is None or not holding.any():
        return step, predicted, multipliers, made[0], active
    owners = elimination.node_of_columns(system, level)
    for member in np.flatnonzero(holding):
        columns = np.flatnonzero(owners == member)
        lower, upper = bounds[0][columns], bounds[1][columns]
        if box.in_bounds(x[columns] + step[columns], lower, upper):
            continue
        stride, _ = box.step_size_to_bound(x[columns], step[columns], lower, upper)
        theta = box.step_back_ratio(gradient[columns], coordinates.v[columns])
        step[columns] *= stride * theta
    predicted = np.where(holding, member_predicted(system, step, coordinates, level), predicted)
    return step, predicted, multipliers, made[0], active


def member_back(
    rows: RowSet,
    own: tuple,
    system,
    x: np.ndarray,
    trial: np.ndarray,
    coordinates: box.Scaling,
    predicted: np.ndarray,
    active: np.ndarray,
    level: int,
    bounds: tuple | None,
) -> tuple:
    """Every member's trial scaled back to the boundary of an inactive own row it crosses.

    The trial is then moved back onto the own rows it violates (``onto_rows``), each
    member's coordinates alone, and kept strictly inside ``bounds`` as the joint trial
    is. Returns ``(trial, scaled, predicted)``: the trial, its step
    in the subproblem's coordinates before that move, and every member's predicted
    decrease, the model's for that step.
    """
    positions, _ = rows.levels[level]
    count, per = positions.shape
    step = trial - x
    owners = elimination.node_of_columns(system, level)
    member_of = np.maximum(owners, 0)
    below = owners >= 0

    def values(fractions: np.ndarray) -> np.ndarray:
        moved = x + np.where(below, fractions[member_of], 0.0) * step
        return rows.values(moved)[positions].reshape(-1)

    groups = np.repeat(np.arange(count), per)
    fractions = _fractions(values, own[0], ~np.asarray(active, dtype=bool), groups, count)
    shorter = fractions < 1.0
    if shorter.any():
        step = np.where(below, fractions[member_of], 1.0) * step
        predicted = np.where(shorter, member_predicted(system, step, coordinates, level), predicted)
    moving = below & (step != 0.0) & ~frozen(system)
    trial = onto_rows(rows, x + step, coordinates.scale, moving, bounds)
    if bounds is not None:
        inside = box.make_strictly_feasible(trial, bounds[0], bounds[1], rstep=0)
        trial = np.where(moving, inside, trial)
    return trial, coordinates.scale * step, predicted


__all__ = ["InequalityRows"]
