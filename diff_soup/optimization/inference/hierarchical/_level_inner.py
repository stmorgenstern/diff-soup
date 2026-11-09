"""The inner solves of a nested fit on a level system: each member's own step and model."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from numbers import Real
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from diff_soup.optimization.inference.hierarchical._level_solve import (
    _ancestor_values,
    _lower_solve,
    _node_sum,
    _secular_search,
    _split,
    _upper_solve,
    scaled_system,
)
from diff_soup.optimization.inference.hierarchical._level_system import (
    LevelSystem,
    _dtype,
    _offsets,
    node_of_columns,
)


class MemberRows(NamedTuple):
    """Equality rows on every member's own coordinates at a level, the same number per member.

    Row ``r`` of member ``n`` holds ``coefficients[n, :, r] . s_n = target[n, r]`` on the
    member's step ``s_n`` at the level, when ``active[n, r]``; an inactive row is not kept,
    so one program solves every active set. The coordinates are those of the system the
    rows are solved with.

    Attributes:
        coefficients: ``(N_level, v_level, R)``.
        target: ``(N_level, R)``.
        active: ``(N_level, R)``, boolean.
    """

    coefficients: ArrayLike
    target: ArrayLike
    active: ArrayLike


@functools.partial(jax.jit, static_argnames=("level",))
def _inner_solve(system, damping, rows=None, *, level):
    """Every member of ``level`` solves its own subtree with its own damping.

    The member's subtree (its block and every level below it) is eliminated
    from the experiments up, as ``_fold`` does, but nothing is folded above
    ``level``: the coordinates there are held. Member ``n`` solves
    ``(H_n + damping_n I) s_n = -g_n``, ``H_n`` and ``g_n`` its subtree's
    blocks, under its active ``rows`` (``MemberRows``, or ``None``): their
    coefficients ride as extra right-hand-side columns, and the step is
    projected onto them as ``_solve`` projects onto the equality rows.
    ``q_n^2 = s_n^T (H_n + damping_n I)^-1 s_n`` comes from the same factors.
    Returns the steps in joint columns (zero above ``level``) and ``q^2`` per
    member.
    """
    counts, widths = system.counts, system.widths
    depth = system.depth
    dtype = system.root.dtype
    offsets = _offsets(counts, widths)
    extra = 0 if rows is None else rows.coefficients.shape[-1]

    def own_damping(at: int) -> jax.Array:
        return damping if at == level else damping[system.ancestors[at][level]]

    diagonal = list(system.diagonal)
    cross = [None] + [list(row) for row in system.cross[1:]]
    rhs: list = [None] * (depth + 1)
    for at in range(level, depth + 1):
        right = -system.gradient[at][..., None]
        if extra:
            columns = (
                rows.coefficients
                if at == level
                else jnp.zeros(right.shape[:2] + (extra,), dtype=dtype)
            )
            right = jnp.concatenate([right, columns], axis=2)
        rhs[at] = right
    factors: list = [None] * (depth + 1)
    couplings: list = [None] * (depth + 1)
    solved: list = [None] * (depth + 1)
    for at in range(depth, level, -1):
        width = widths[at]
        if width == 0:
            continue
        factor = jax.vmap(jnp.linalg.cholesky)(
            diagonal[at] + own_damping(at)[:, None, None] * jnp.eye(width, dtype=dtype)
        )
        factors[at] = factor
        row: list = [None] * at
        for above in range(level, at):
            if widths[above]:
                row[above] = _lower_solve(factor, cross[at][above])
        couplings[at] = row
        solved[at] = _lower_solve(factor, rhs[at])
        for above in range(level, at):
            if not widths[above]:
                continue
            for other in range(level, above + 1):
                if not widths[other]:
                    continue
                product = jnp.einsum("nia,nib->nab", row[above], row[other])
                total = _node_sum(system, product, at, above)
                if other == above:
                    diagonal[above] = diagonal[above] - total
                else:
                    cross[above][other] = cross[above][other] - total
            folded = jnp.einsum("nia,nik->nak", row[above], solved[at])
            rhs[above] = rhs[above] - _node_sum(system, folded, at, above)
    steps: list = [None] * (depth + 1)
    if widths[level]:
        factor = jax.vmap(jnp.linalg.cholesky)(
            diagonal[level] + damping[:, None, None] * jnp.eye(widths[level], dtype=dtype)
        )
        factors[level] = factor
        solved[level] = _lower_solve(factor, rhs[level])
        steps[level] = _upper_solve(factor, solved[level])
    else:
        steps[level] = jnp.zeros((counts[level - 1], 0, 1 + extra), dtype=dtype)
    for at in range(level + 1, depth + 1):
        if not widths[at]:
            steps[at] = jnp.zeros((counts[at - 1], 0, 1 + extra), dtype=dtype)
            continue
        right = solved[at]
        for above in range(level, at):
            if not widths[above]:
                continue
            above_steps = _ancestor_values(system, steps[above], at, above)
            right = right - jnp.einsum("nia,nak->nik", couplings[at][above], above_steps)
        steps[at] = _upper_solve(factors[at], right)
    if extra:
        # Every member's multipliers: (C_n Y_n) mu_n = C_n s_n - target_n on its active
        # rows, an inactive row's row and column the identity's (a zero multiplier).
        coefficients, on = rows.coefficients, rows.active
        gram = jnp.einsum("nir,nis->nrs", coefficients, steps[level][..., 1:])
        residual = jnp.einsum("nir,ni->nr", coefficients, steps[level][..., 0]) - rows.target
        gram = jnp.where(on[:, :, None] & on[:, None, :], gram, 0.0) + jax.vmap(jnp.diag)(
            jnp.where(on, 0.0, 1.0)
        )
        multipliers = jnp.linalg.solve(gram, jnp.where(on, residual, 0.0)[..., None])[..., 0]
        for at in range(level, depth + 1):
            own = multipliers if at == level else multipliers[system.ancestors[at][level]]
            steps[at] = steps[at][..., 0] - jnp.einsum("nik,nk->ni", steps[at][..., 1:], own)
    else:
        steps = steps[:level] + [entry[..., 0] for entry in steps[level:]]
    halves: list = [None] * (depth + 1)
    q_squared = jnp.zeros((counts[level - 1],), dtype=dtype)
    for at in range(depth, level - 1, -1):
        if not widths[at]:
            continue
        right = steps[at]
        for deeper in range(at + 1, depth + 1):
            if not widths[deeper]:
                continue
            carried = jnp.einsum("nia,ni->na", couplings[deeper][at], halves[deeper])
            right = right - _node_sum(system, carried, deeper, at)
        halves[at] = _lower_solve(factors[at], right[..., None])[..., 0]
        squares = jnp.sum(halves[at] * halves[at], axis=1)
        q_squared = q_squared + (squares if at == level else _node_sum(system, squares, at, level))
    step = jnp.concatenate(
        [jnp.zeros((offsets[level],), dtype=dtype)]
        + [steps[at].reshape(-1) for at in range(level, depth + 1)]
    )
    return step, q_squared


@functools.partial(jax.jit, static_argnames=("level",))
def _node_model(system, delta, *, level):
    """Per member of ``level``: ``g_n . delta_n`` and ``delta_n^T H_n delta_n`` over its subtree."""
    parts = _split(system, delta)
    count = system.counts[level - 1]
    dtype = system.root.dtype
    linear = jnp.zeros((count,), dtype=dtype)
    quadratic = jnp.zeros((count,), dtype=dtype)
    for at in range(level, system.depth + 1):
        if not system.widths[at]:
            continue
        own = parts[at]
        slope = jnp.einsum("ni,ni->n", own, system.gradient[at])
        curvature = jnp.einsum("ni,nij,nj->n", own, system.diagonal[at], own)
        for above in range(level, at):
            if not system.widths[above]:
                continue
            other = _ancestor_values(system, parts[above], at, above)
            curvature = curvature + 2.0 * jnp.einsum(
                "ni,nij,nj->n", own, system.cross[at][above], other
            )
        if at == level:
            linear = linear + slope
            quadratic = quadratic + curvature
        else:
            linear = linear + _node_sum(system, slope, at, level)
            quadratic = quadratic + _node_sum(system, curvature, at, level)
    return linear, quadratic


def _scaled_rows(
    system: LevelSystem, rows: MemberRows | None, scale: np.ndarray, level: int
) -> MemberRows | None:
    """``rows`` in the coordinates ``z = scale * s``: each coefficient over its column's scale."""
    if rows is None:
        return None
    dtype = _dtype(system)
    offsets = _offsets(system.counts, system.widths)
    own = scale[offsets[level] : offsets[level + 1]].reshape(
        system.counts[level - 1], system.widths[level]
    )
    with np.errstate(all="ignore"):
        coefficients = np.asarray(rows.coefficients, dtype=dtype) / own[:, :, None]
    return MemberRows(
        jnp.asarray(coefficients),
        jnp.asarray(np.asarray(rows.target, dtype=dtype)),
        jnp.asarray(np.asarray(rows.active, dtype=bool)),
    )


def inner_solve_scaled(
    system: LevelSystem,
    scale: ArrayLike | Sequence[Real],
    level: int,
    *,
    addition: ArrayLike | Sequence | None = None,
    rows: MemberRows | None = None,
) -> tuple:
    """Every member of ``level`` solves its subtree at zero damping, in ``z = scale * s``.

    The coordinates above ``level`` are held. This is the solve the members'
    multiplier searches of ``inner_secular_step`` make first, for every
    member at once.

    Args:
        system: The system.
        scale: ``(n,)``, positive.
        level: The level of the inner problems, ``1 .. D``.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix (``scaled_system``).
        rows: Optional ``MemberRows`` on the members' own coordinates at
            ``level``, in the coordinates of ``system``: every member's step
            keeps its active rows.

    Returns:
        ``(z, q_squared)`` as host arrays: the scaled steps ``(n,)`` in joint
        columns (zero above ``level``) and ``z_n^T H_n^-1 z_n`` per member
        ``(N_level,)``, ``H_n`` the member's scaled subtree matrix. A member
        whose matrix has no finite factor gets non-finite values.
    """
    dtype = _dtype(system)
    scale = np.asarray(scale, dtype=dtype)
    scaled = scaled_system(system, scale, addition)
    count = system.counts[int(level) - 1]
    held = _scaled_rows(system, rows, scale, int(level))
    step, q_squared = _inner_solve(scaled, np.zeros((count,), dtype=dtype), held, level=int(level))
    return np.asarray(step), np.asarray(q_squared)


def _lockstep_search(
    scaled: LevelSystem,
    level: int,
    radii: ArrayLike,
    active: ArrayLike,
    first: tuple[ArrayLike, ArrayLike] | None,
    *,
    hold: bool = False,
    rows: MemberRows | None = None,
) -> tuple:
    """The multiplier searches of the active members of ``level`` on a scaled system.

    Every active member runs ``_secular_search`` with its own radius on its
    subtree of ``scaled``, the coordinates above ``level`` held. The members
    search in lockstep: every round is one damped solve of all subtrees at
    once, each with its own damping. ``first`` is an optional
    ``(z, q_squared)`` of the solve at zero damping, already made; it answers
    every search's first request and counts as one solve. With ``hold``, a
    member whose search finds no finite step inside its radius takes a zero
    step and a ``nan`` multiplier instead of raising. ``rows`` are the
    members' ``MemberRows`` in the scaled coordinates, kept by every solve.

    Returns:
        ``(multipliers, scaled_step, solves)``: the multipliers
        ``(N_level,)``, the scaled steps ``(n,)`` in joint columns (zero above
        ``level`` and on inactive members), and the number of solves made.
    """
    dtype = _dtype(scaled)
    owners = node_of_columns(scaled, level)
    count = scaled.counts[level - 1]
    columns = [np.flatnonzero(owners == member) for member in range(count)]
    eps = float(np.finfo(dtype).eps)
    tiny = float(np.finfo(float).tiny)
    searches = {
        int(member): _secular_search(float(radii[member]), eps, tiny)
        for member in np.flatnonzero(active)
    }
    requests = {member: next(search) for member, search in searches.items()}
    multipliers = np.zeros((count,), dtype=np.float64)
    scaled_step = np.zeros((owners.shape[0],), dtype=dtype)
    solves = 0

    def answer(requests: dict, step: np.ndarray, q_squared: np.ndarray) -> dict:
        """Send every requesting member its solve; the requests of the next round."""
        following = {}
        for member in requests:
            try:
                following[member] = searches[member].send(
                    (step[columns[member]], float(q_squared[member]))
                )
            except StopIteration as found:
                multiplier, member_step = found.value
                multipliers[member] = multiplier
                scaled_step[columns[member]] = member_step
            except ValueError:
                if not hold:
                    raise
                multipliers[member] = np.nan
        return following

    if first is not None and requests:
        # Every search asks for zero damping first; the solve already made answers it.
        solves += 1
        requests = answer(requests, np.asarray(first[0]), np.asarray(first[1]))
    while requests:
        damping = np.ones((count,), dtype=dtype)
        for member, value in requests.items():
            damping[member] = value
        step, q_squared = _inner_solve(scaled, damping, rows, level=level)
        solves += 1
        requests = answer(requests, np.asarray(step), np.asarray(q_squared))
    return multipliers, scaled_step, solves


def inner_secular_step(
    system: LevelSystem,
    scale: ArrayLike | Sequence[Real],
    radii: ArrayLike,
    level: int,
    active: ArrayLike,
    *,
    first: tuple[ArrayLike, ArrayLike] | None = None,
    addition: ArrayLike | Sequence | None = None,
    hold: bool = False,
    rows: MemberRows | None = None,
) -> tuple:
    """The trust-region step of every active member of ``level`` over its own subtree.

    Each active member runs the multiplier search of ``level_secular_step``
    with its own radius in the scaled norm ``||scale * s_n||``, the
    coordinates above ``level`` held. The members search in lockstep: every
    round is one damped solve of all subtrees at once, each with its own
    multiplier.

    Args:
        system: The system.
        scale: ``(n,)``, positive.
        radii: ``(N_level,)``, the radius of every member.
        level: The level of the inner problems, ``1 .. D``.
        active: ``(N_level,)`` bool; an inactive member takes a zero step.
        first: Optional ``(z, q_squared)`` of ``inner_solve_scaled`` with
            this scale and level, already made: every active member's search
            takes its solve at zero damping from it, and it counts as this
            step's first solve.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix (``scaled_system``); every member's model carries it.
        hold: A member for which no finite step inside its radius is found
            takes a zero step, a zero predicted decrease and a ``nan``
            multiplier instead of raising; the other members are unchanged.
        rows: Optional ``MemberRows`` on the members' own coordinates at
            ``level``, in the coordinates of ``system``: every member's step
            keeps its active rows. ``first`` must have been solved with them.

    Returns:
        ``(step, predicted, multipliers, solves)``: the step ``(n,)`` in joint
        columns (zero above ``level`` and on inactive members), the decrease
        each member's model predicts ``(N_level,)`` (zero when inactive), the
        multipliers ``(N_level,)``, and the number of solves made.

    Raises:
        ValueError: If no finite step inside its radius is found for a member
            and ``hold`` is false.
    """
    dtype = _dtype(system)
    level = int(level)
    scale = np.asarray(scale, dtype=dtype)
    radii = np.asarray(radii, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    scaled = scaled_system(system, scale, addition)
    multipliers, scaled_step, solves = _lockstep_search(
        scaled,
        level,
        radii,
        active,
        first,
        hold=hold,
        rows=_scaled_rows(system, rows, scale, level),
    )
    with np.errstate(all="ignore"):
        delta = scaled_step / scale
    predicted = inner_predicted(system, delta, level)
    if addition is not None:
        predicted -= 0.5 * inner_added(system, addition, scaled_step, level)
    predicted[~active | np.isnan(multipliers)] = 0.0
    return delta, predicted, multipliers, solves


def inner_predicted(
    system: LevelSystem, delta: ArrayLike | Sequence[Real], level: int
) -> np.ndarray:
    """``-(g_n . delta_n + delta_n^T H_n delta_n / 2)`` for every member ``n`` of ``level``.

    ``H_n`` and ``g_n`` are the blocks and the gradient of the member's subtree;
    the columns of ``delta`` above ``level`` do not enter.
    """
    delta = np.asarray(delta, dtype=_dtype(system))
    linear, quadratic = _node_model(system, delta, level=int(level))
    return -(np.asarray(linear, dtype=np.float64) + 0.5 * np.asarray(quadratic, dtype=np.float64))


def model_split(system: LevelSystem, delta: ArrayLike | Sequence[Real], level: int) -> tuple:
    """The decrease the Gauss-Newton model predicts for ``delta``, split by member of ``level``.

    Returns ``(members, rest)``: ``members`` ``(N_level,)``, for every member ``n``
    its subtree's terms, ``-(g_n . delta_n + delta_n^T H_n delta_n / 2 +
    delta_n^T C_n delta_a)`` with ``C_n`` the coupling of the subtree to the
    coordinates above ``level`` and ``delta_a`` their step; ``rest`` the terms
    of the coordinates above ``level`` alone. ``members.sum() + rest`` is the
    model's decrease ``-(g . delta + delta^T H delta / 2)``.
    """
    level = int(level)
    delta = np.asarray(delta, dtype=_dtype(system))
    parts = _split(system, delta)
    count = system.counts[level - 1]
    coupling = np.zeros((count,), dtype=np.float64)
    full = float(parts[0] @ np.asarray(system.root_gradient)) + 0.5 * float(
        parts[0] @ np.asarray(system.root) @ parts[0]
    )
    for at in range(1, system.depth + 1):
        if not system.widths[at]:
            continue
        own = np.asarray(parts[at])
        terms = np.einsum("ni,ni->n", own, np.asarray(system.gradient[at])) + 0.5 * np.einsum(
            "ni,nij,nj->n", own, np.asarray(system.diagonal[at]), own
        )
        for above in range(0, at):
            if not system.widths[above]:
                continue
            if above == 0:
                other = np.broadcast_to(np.asarray(parts[0]), (own.shape[0], system.widths[0]))
            else:
                other = np.asarray(_ancestor_values(system, parts[above], at, above))
            pair = np.einsum("ni,nij,nj->n", own, np.asarray(system.cross[at][above]), other)
            terms = terms + pair
            if at >= level and above < level:
                coupling += np.asarray(
                    pair if at == level else _node_sum(system, pair, at, level), dtype=np.float64
                )
        full += float(np.sum(terms))
    members = inner_predicted(system, delta, level) - coupling
    return members, float(-full - np.sum(members))


def inner_quadratic(
    system: LevelSystem, delta: ArrayLike | Sequence[Real], level: int
) -> np.ndarray:
    """``delta_n^T H_n delta_n`` for every member ``n`` of ``level``, ``(N_level,)``.

    ``H_n`` is the matrix of the member's subtree; the columns of ``delta``
    above ``level`` do not enter.
    """
    delta = np.asarray(delta, dtype=_dtype(system))
    _, quadratic = _node_model(system, delta, level=int(level))
    return np.asarray(quadratic, dtype=np.float64)


def inner_added(
    system: LevelSystem, addition: ArrayLike | Sequence | None, scaled_step: ArrayLike, level: int
) -> np.ndarray:
    """``z_n^T diag(addition_n) z_n`` for every member ``n`` of ``level``, ``(N_level,)``.

    ``z`` is a scaled step in joint columns; the columns above ``level`` do
    not enter.
    """
    level = int(level)
    owners = node_of_columns(system, level)
    below = owners >= 0
    with np.errstate(all="ignore"):
        terms = np.asarray(addition, dtype=np.float64) * np.square(
            np.asarray(scaled_step, dtype=np.float64)
        )
    return np.bincount(owners[below], weights=terms[below], minlength=system.counts[level - 1])


def inner_programs(system: LevelSystem, level: int) -> None:
    """Compile the programs of the inner problems of ``level`` before a fit's loop.

    These are the members' damped solves and their models.
    """
    level = int(level)
    count = system.counts[level - 1]
    dtype = _dtype(system)
    scaled = scaled_system(system, np.ones((system.size,), dtype=dtype))
    _inner_solve(scaled, np.ones((count,), dtype=dtype), level=level)
    _node_model(system, np.zeros((system.size,), dtype=dtype), level=level)


__all__ = [
    "MemberRows",
    "inner_added",
    "inner_predicted",
    "inner_programs",
    "inner_quadratic",
    "inner_secular_step",
    "inner_solve_scaled",
    "model_split",
]
