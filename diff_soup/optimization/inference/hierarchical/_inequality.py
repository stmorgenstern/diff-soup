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
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup.optimization.inference.hierarchical import _bounds as box
from diff_soup.optimization.inference.hierarchical import _level_elimination as elimination
from diff_soup.optimization.inference.hierarchical._active_set import active_set, slack
from diff_soup.optimization.inference.hierarchical._level_inner import (
    _inner_multipliers,
    _inner_solve,
    _inner_step,
    _member_predicted,
)
from diff_soup.optimization.inference.hierarchical._level_solve import _scaled
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


# ---------------------------------------------------------------------------
# The rows inside a compiled program
# ---------------------------------------------------------------------------


def frozen_columns(system: elimination.LevelSystem) -> jax.Array:
    """``(n,)``: the coordinates a step of ``system`` never moves, fixed or of a held node."""
    held = [jnp.zeros((system.widths[0],), dtype=bool)] + [
        jnp.repeat(jnp.asarray(system.held_mask[level], dtype=bool), system.widths[level])
        for level in range(1, system.depth + 1)
    ]
    return jnp.asarray(system.fixed_mask, dtype=bool) | jnp.concatenate(held)


def traced_rows(rows: RowSet, x: jax.Array, jacobian: bool = True) -> tuple:
    """``(c, C)`` at a traced ``x``, as ``RowSet.__call__`` gives them on the host (``C`` ``None`` without ``jacobian``)."""
    shared = x[: rows.shared]
    values, jacobians = [], []
    for program, columns, data in rows.blocks:
        if columns is None:
            c, d_shared = program(shared, data)
            values.append(c)
            if jacobian:
                block = jnp.zeros((c.shape[0], rows.n), dtype=x.dtype)
                jacobians.append(block.at[:, : rows.shared].set(d_shared))
            continue
        c, d_own, d_shared = program(x[columns], shared, data)
        members, count = c.shape
        values.append(c.reshape(-1))
        if jacobian:
            block = jnp.zeros((members, count, rows.n), dtype=x.dtype)
            block = block.at[:, :, : rows.shared].set(d_shared)
            block = block.at[
                np.arange(members)[:, None, None],
                np.arange(count)[None, :, None],
                np.asarray(columns)[:, None, :],
            ].set(d_own)
            jacobians.append(block.reshape(members * count, rows.n))
    c = jnp.concatenate(values) if values else jnp.zeros((0,), dtype=x.dtype)
    if not jacobian:
        return c, None
    C = jnp.concatenate(jacobians) if jacobians else jnp.zeros((0, rows.n), dtype=x.dtype)
    if rows.fixed.size:
        C = C.at[:, rows.fixed].set(0.0)
    return c, C


class Placed(NamedTuple):
    """Every row of a ``RowSet`` placed in a traced system at a fixed shape, all inactive.

    ``template`` is the system holding every row placed,
    ``keep`` ``(R,)`` the rows with a column a step moves, the only ones ever kept, and
    ``general`` the static positions of the rows of no member among ``rows``.
    """

    template: elimination.LevelSystem
    keep: jax.Array
    general: np.ndarray


def place_rows(
    rows: RowSet, system: elimination.LevelSystem, x: jax.Array, c: jax.Array, C: jax.Array
) -> Placed:
    """Every row of ``rows`` placed in ``system`` at a fixed shape for a compiled program, none active.

    A member's rows ride at its node (``member_rows``, its ``MemberRows`` with every row's
    coefficients on its own columns and its ancestors'), the others among the equality rows;
    row ``i`` reads ``C_i (x + s) = C_i x - c_i``, the step on which its linearization at ``x``
    is zero. ``C`` has the columns a step never moves set to zero; a row with no column left
    is never kept. Flip the rows kept with ``hold_rows``; one program solves every active
    set.
    """
    dtype = system.root.dtype
    keep = jnp.any(C != 0.0, axis=1)
    member_rows: list = [None] * (system.depth + 1)
    for level in _member_levels(rows, system):
        member_rows[level] = _place_traced(rows, system, c, C, level)
    general = _general_rows(rows, system)
    fields: dict = {}
    if general.size:
        count = int(np.shape(system.constraint_matrix)[0])
        kept = (
            jnp.ones((count,), dtype=bool)
            if system.constraint_active is None
            else jnp.asarray(system.constraint_active, dtype=bool)
        )
        fields = {
            "constraint_matrix": jnp.concatenate(
                [system.constraint_matrix, C[general].astype(dtype)]
            ),
            "constraint_rhs": jnp.concatenate(
                [system.constraint_rhs, (C[general] @ x - c[general]).astype(dtype)]
            ),
            "constraint_active": jnp.concatenate([kept, jnp.zeros((general.size,), dtype=bool)]),
        }
    placed = tuple(member_rows) if any(entry is not None for entry in member_rows) else None
    names = ("member_rows",) + tuple(fields)
    values = (placed,) + tuple(fields.values())
    template = eqx.tree_at(
        lambda s: tuple(getattr(s, name) for name in names),
        system,
        values,
        is_leaf=lambda leaf: leaf is None,
    )
    return Placed(template, keep, general)


def _place_traced(rows: RowSet, system, c: jax.Array, C: jax.Array, level: int):
    """The members' rows of ``level`` as ``MemberRows``: their coefficients from ``C`` per node, target ``-c``.

    Row ``i`` holds ``C_i s = -c_i``, the step on which its linearization at the point of
    ``c`` and ``C`` is zero; ``C`` is read on each member's own columns and on those of its
    ancestors. Every row is inactive.
    """
    dtype = system.root.dtype
    positions, columns = rows.levels[level]
    count, per = positions.shape
    offsets = _offsets(system.counts, system.widths)
    block = C[positions]
    nodes = np.arange(count)[:, None, None]
    numbers = np.arange(per)[None, :, None]

    def read(read_columns) -> jax.Array:
        return jnp.transpose(block[nodes, numbers, read_columns[:, None, :]], (0, 2, 1)).astype(
            dtype
        )

    above = []
    for at in range(level):
        width = system.widths[at]
        if at == 0:
            node_columns = np.broadcast_to(np.arange(width), (count, width))
        else:
            ancestor = jnp.asarray(system.ancestors[level][at])
            node_columns = offsets[at] + ancestor[:, None] * width + jnp.arange(width)[None, :]
        above.append(read(node_columns))
    return elimination.MemberRows(
        coefficients=read(columns),
        above=tuple(above),
        target=(-c[positions]).astype(dtype),
        active=jnp.zeros((count, per), dtype=bool),
    )


def hold_rows(rows: RowSet, placed: Placed, active: jax.Array) -> elimination.LevelSystem:
    """The placed system with the rows ``active`` (among those with a column a step moves) kept."""
    on = active & placed.keep
    template = placed.template
    member_rows = template.member_rows
    if member_rows is not None:
        member_rows = tuple(
            None if entry is None else entry._replace(active=on[rows.levels[level][0]])
            for level, entry in enumerate(member_rows)
        )
    if not placed.general.size:
        return eqx.tree_at(
            lambda s: s.member_rows, template, member_rows, is_leaf=lambda leaf: leaf is None
        )
    count = int(np.shape(template.constraint_matrix)[0]) - placed.general.size
    flags = jnp.concatenate([template.constraint_active[:count], on[placed.general]])
    return eqx.tree_at(
        lambda s: (s.member_rows, s.constraint_active),
        template,
        (member_rows, flags),
        is_leaf=lambda leaf: leaf is None,
    )


def rows_multipliers(rows: RowSet, system, placed: Placed, flat: jax.Array) -> jax.Array:
    """``(R,)``: the held system's flat multipliers in the order of ``rows``, zero on a row not kept."""
    multipliers = jnp.zeros((rows.count,), dtype=flat.dtype)
    start = int(np.shape(system.constraint_matrix)[0])
    stop = start + placed.general.size
    if placed.general.size:
        multipliers = multipliers.at[placed.general].set(flat[start:stop])
    for level in _member_levels(rows, system):
        positions = rows.levels[level][0].reshape(-1)
        multipliers = multipliers.at[positions].set(flat[stop : stop + positions.size])
        stop += positions.size
    return multipliers


def traced_fractions(values: Callable, c0, watched, group_of: np.ndarray, count: int):
    """Per group, the largest fraction of its step keeping its watched rows within ``max(c0, 0)``.

    ``values(fractions)`` gives the rows at the point moved by every group's fraction of
    its step; ``group_of[i]`` is the group of row ``i``, which reads its group's fraction
    alone. A bisection of every group at once, the rows being nonlinear: a group whose step
    crosses a row is scaled back to that row's boundary.
    """
    limit = jnp.maximum(c0, 0.0) + slack(c0, jnp.zeros((c0.size, 1)), jnp.zeros((1,)))

    def broken(fractions):
        crossed = watched & ~(values(fractions) <= limit)
        return jnp.zeros((count,), dtype=bool).at[group_of].max(crossed)

    high = jnp.ones((count,), dtype=c0.dtype)
    low = jnp.where(broken(high), 0.0, 1.0).astype(c0.dtype)

    def bisect(pair):
        def halve(_, pair):
            low, high = pair
            middle = jnp.where(low < 1.0, 0.5 * (low + high), 1.0)
            bad = broken(middle)
            return jnp.where(bad, low, middle), jnp.where(bad, middle, high)

        return jax.lax.fori_loop(0, _STEP_BACK_BISECTIONS, halve, pair)

    low, _ = jax.lax.cond(jnp.all(low == 1.0), lambda pair: pair, bisect, (low, high))
    return low


def traced_step_back(rows: RowSet, x, step, active):
    """The largest fraction of ``step`` at which no row outside ``active`` is worse than ``max(c(x), 0)``.

    1 when the full step keeps those rows. An active row, held at its boundary by the
    step's linearization, keeps the step's second-order residual, which the next step's
    projection takes back. A bisection in the compiled program.
    """
    c0 = traced_rows(rows, x, jacobian=False)[0]
    fraction = traced_fractions(
        lambda t: traced_rows(rows, x + t[0] * step, jacobian=False)[0],
        c0,
        ~active,
        np.zeros((rows.count,), dtype=np.int64),
        1,
    )
    return fraction[0]


def traced_onto_rows(rows: RowSet, x, scale, moving, bounds=None):
    """``x`` moved back onto the rows it violates beyond their slack, the coordinates ``moving`` alone.

    A step that holds a curved row on its linearization leaves it by the square of the
    step; the Newton projection ``x - D^-2 C^T (C D^-2 C^T)^-1 c`` onto the violated rows,
    in the metric ``D = diag(scale)`` of the step, takes the trial back to their boundary,
    a few times at most. A row the coordinates ``moving`` do not reach is left as it is.
    Under ``bounds`` a coordinate the projection takes out of the box is put back strictly
    inside it and held, and the projection goes on with the others. The projection solves
    ``(A A^T) mu = c`` on the violated rows at a fixed shape (a row not violated takes the
    identity's row), in the dtype of ``x``, as every solve of the step.
    """
    zeros_rows = jnp.zeros((rows.count, 1), dtype=x.dtype)

    def body(state):
        x, moving, left, _ = state
        c, C = traced_rows(rows, x)
        A = jnp.where(moving[None, :], C, 0.0) / scale[None, :]
        A = jnp.where(jnp.isfinite(A), A, 0.0)
        reached = jnp.any(A != 0.0, axis=1)
        violated = reached & (c > slack(c, zeros_rows, jnp.zeros((1,), dtype=x.dtype)))
        A = A * violated[:, None]
        gram = A @ A.T + jnp.diag(jnp.where(violated, 0.0, 1.0).astype(x.dtype))
        mu = jnp.linalg.solve(gram, jnp.where(violated, c, 0.0))
        moved = x - jnp.where(moving, (A.T @ mu) / scale, 0.0)
        left_next = left - 1
        moving_next = moving
        if bounds is not None:
            lower, upper = bounds
            out = moving & ~((moved > lower) & (moved < upper))
            moved = jnp.where(out, box.make_strictly_feasible(moved, lower, upper, rstep=0), moved)
            moving_next = moving & ~out
            left_next = jnp.where(jnp.any(out), _RESTORE_ITERATIONS, left_next)
        stop = ~jnp.any(violated)
        return (
            jnp.where(stop, x, moved),
            jnp.where(stop, moving, moving_next),
            jnp.where(stop, left, left_next),
            stop,
        )

    state = (
        x,
        jnp.asarray(moving, dtype=bool),
        jnp.asarray(_RESTORE_ITERATIONS),
        jnp.asarray(False),
    )
    x, _, _, _ = jax.lax.while_loop(lambda state: (state[2] > 0) & ~state[3], body, state)
    return x


# ---------------------------------------------------------------------------
# A member's own rows in the polish, in a compiled program
# ---------------------------------------------------------------------------


class MemberRowSet(NamedTuple):
    """The rows of the members of a level on their own values at a point, placed for the polish.

    ``c`` ``(N P,)`` and ``C`` ``(N P, n)`` are the rows and their Jacobian member by
    member, ``C`` zero on the columns a step never moves and on those above the level,
    which the polish holds; ``template`` is the system holding every row at its member's
    node, none active; ``keep`` ``(N P,)`` the rows that read a column of their member a
    step moves, the only ones ever held.
    """

    c: jax.Array
    C: jax.Array
    template: elimination.LevelSystem
    keep: jax.Array


class _MemberZero(NamedTuple):
    step: jax.Array
    z: jax.Array
    q_squared: jax.Array
    solves: jax.Array
    failed: jax.Array


class _MemberStep(NamedTuple):
    step: jax.Array
    predicted: jax.Array
    multipliers: jax.Array
    solves: jax.Array
    failed: jax.Array


def member_row_set(
    rows: RowSet, system: elimination.LevelSystem, x: jax.Array, level: int, owners: np.ndarray
) -> MemberRowSet | None:
    """The rows of the members of ``level`` on their own values at ``x``; ``None`` when the level has none.

    ``owners`` is ``node_of_columns(system, level)``, static. Every row is placed at its
    member's node at a fixed shape (``_place_traced``), so one program solves every active
    set; the levels' other rows are not held by the polish.
    """
    if level not in rows.levels or not system.widths[level]:
        return None
    positions = rows.levels[level][0]
    count, per = positions.shape
    c, C = traced_rows(rows, x)
    C = jnp.where(frozen_columns(system)[None, :], 0.0, C)
    entry = _place_traced(rows, system, c, C, level)
    member_rows: list = [None] * (system.depth + 1)
    member_rows[level] = entry
    template = eqx.tree_at(
        lambda s: s.member_rows, system, tuple(member_rows), is_leaf=lambda leaf: leaf is None
    )
    own = jnp.where(jnp.asarray(owners >= 0)[None, :], C[positions].reshape(count * per, -1), 0.0)
    keep = jnp.any(entry.coefficients != 0.0, axis=1).reshape(-1)
    return MemberRowSet(c[positions].reshape(-1), own, template, keep)


def _hold_members(own: MemberRowSet, level: int, active: jax.Array) -> elimination.LevelSystem:
    """The placed system with the members' rows ``active`` (among those ``own.keep``) held."""
    entry = own.template.member_rows[level]
    member_rows = list(own.template.member_rows)
    member_rows[level] = entry._replace(active=(active & own.keep).reshape(entry.active.shape))
    return eqx.tree_at(
        lambda s: s.member_rows, own.template, tuple(member_rows), is_leaf=lambda leaf: leaf is None
    )


def _addition_of(coordinates: box.Scaling) -> jax.Array:
    return (
        jnp.zeros_like(coordinates.scale) if coordinates.addition is None else coordinates.addition
    )


def member_zero(
    own: MemberRowSet, coordinates: box.Scaling, x: jax.Array, level: int, owners: np.ndarray
) -> tuple:
    """Every member's solve at zero damping under its own rows, by the active-set method.

    Every member is a group of the method (``active_set``), its rows read its own
    columns alone. Returns ``(zero, first, solves)``: the solve ``(z, q_squared)``, the
    solve with no row held (the first solve of the members' searches), and the solves made
    beyond that one. A member whose active set did not settle has the method's last
    iterate as its ``z`` and ``q_squared`` ``nan``: its model's decrease is not known.
    """
    scale, addition = coordinates.scale, _addition_of(coordinates)
    count, _, per = own.template.member_rows[level].coefficients.shape
    damping = jnp.zeros((count,), dtype=scale.dtype)

    def solve(active):
        held = _scaled(_hold_members(own, level, active), scale, addition)
        z, q_squared = _inner_solve(held, damping, level=level)
        return _MemberZero(z / scale, z, q_squared, jnp.asarray(1), jnp.asarray(False))

    def multipliers(active, answer):
        held = _scaled(_hold_members(own, level, active), scale, addition)
        return _inner_multipliers(held, damping, level=level).reshape(-1)

    first = solve(jnp.zeros((count * per,), dtype=bool))
    found = active_set(
        solve, multipliers, own.c, own.C, x, first, shape=(count, per), columns=owners
    )
    unsettled = jnp.asarray(owners >= 0) & ~found.settled[np.maximum(owners, 0)]
    z = jnp.where(unsettled, scale * found.step, found.answer.z)
    q_squared = jnp.where(found.settled, found.answer.q_squared, jnp.nan)
    return (z, q_squared), (first.z, first.q_squared), found.solves


def member_step(
    own: MemberRowSet,
    system: elimination.LevelSystem,
    coordinates: box.Scaling,
    x: jax.Array,
    radii: jax.Array,
    members: jax.Array,
    first: tuple,
    level: int,
    owners: np.ndarray,
    *,
    eps: float,
    tiny: float,
) -> tuple:
    """The members' trust-region steps under their own rows, by the active-set method.

    Every round solves every member's step with its held rows (``_inner_step``); a round
    with no row held starts its searches from ``first``, the solve holding none. A member
    whose active set does not settle takes the method's last iterate, a descent step inside
    its radius that keeps its rows, its predicted decrease the model's. Returns ``(step,
    predicted, multipliers, solves, active, holding)``: ``active`` the rows held and
    ``holding`` the members ``members`` that hold one.
    """
    scale, addition = coordinates.scale, _addition_of(coordinates)
    count, _, per = own.template.member_rows[level].coefficients.shape
    damping = jnp.zeros((count,), dtype=scale.dtype)
    traced_owners = jnp.asarray(owners)

    def solve(active):
        held = _hold_members(own, level, active)
        start = jax.lax.cond(
            jnp.any(active & own.keep),
            lambda: _inner_solve(_scaled(held, scale, addition), damping, level=level),
            lambda: tuple(first),
        )
        delta, predicted, multipliers, solves, _ = _inner_step(
            held,
            scale,
            radii,
            members,
            addition,
            traced_owners,
            start,
            level=level,
            eps=eps,
            tiny=tiny,
        )
        return _MemberStep(delta, predicted, multipliers, solves, jnp.asarray(False))

    def multipliers(active, answer):
        held = _scaled(_hold_members(own, level, active), scale, addition)
        return _inner_multipliers(held, jnp.nan_to_num(answer.multipliers), level=level).reshape(-1)

    found = active_set(
        solve, multipliers, own.c, own.C, x, None, shape=(count, per), columns=owners
    )
    unsettled = jnp.asarray(owners >= 0) & ~found.settled[np.maximum(owners, 0)]
    step = jnp.where(unsettled, found.step, found.answer.step)
    multipliers_ = found.answer.multipliers
    model = _member_predicted(
        system, step, scale * step, addition, traced_owners, count, level=level
    )
    predicted = jnp.where(found.settled, found.answer.predicted, model)
    predicted = jnp.where(~members | jnp.isnan(multipliers_), 0.0, predicted)
    holding = jnp.any(found.active.reshape(count, per), axis=1) & members
    return step, predicted, multipliers_, found.solves, found.active, holding


def member_back(
    rows: RowSet,
    own: MemberRowSet,
    system: elimination.LevelSystem,
    x: jax.Array,
    trial: jax.Array,
    coordinates: box.Scaling,
    predicted: jax.Array,
    active: jax.Array,
    level: int,
    owners: np.ndarray,
    bounds: tuple | None,
) -> tuple:
    """Every member's trial scaled back to the boundary of an inactive own row it crosses.

    The trial is then moved back onto the own rows it violates (``traced_onto_rows``),
    each member's coordinates alone, and kept strictly inside ``bounds`` as the joint
    trial is. Returns ``(trial, scaled, predicted)``: the trial, its step in the
    subproblem's coordinates before that move, and every member's predicted decrease,
    the model's for that step.
    """
    positions = rows.levels[level][0]
    count, per = positions.shape
    scale, addition = coordinates.scale, _addition_of(coordinates)
    owned = jnp.asarray(owners >= 0)
    member_of = np.maximum(owners, 0)
    step = trial - x

    def values(fractions):
        moved = x + jnp.where(owned, fractions[member_of], 0.0) * step
        return traced_rows(rows, moved, jacobian=False)[0][positions].reshape(-1)

    fractions = traced_fractions(values, own.c, ~active, np.repeat(np.arange(count), per), count)
    step = jnp.where(owned, fractions[member_of], 1.0) * step
    model = _member_predicted(
        system, step, scale * step, addition, jnp.asarray(owners), count, level=level
    )
    predicted = jnp.where(fractions < 1.0, model, predicted)
    moving = owned & (step != 0.0) & ~frozen_columns(system)
    moved = traced_onto_rows(rows, x + step, scale, moving, bounds)
    if bounds is not None:
        inside = box.make_strictly_feasible(moved, bounds[0], bounds[1], rstep=0)
        moved = jnp.where(moving, inside, moved)
    return moved, scale * step, predicted


__all__ = ["InequalityRows"]
