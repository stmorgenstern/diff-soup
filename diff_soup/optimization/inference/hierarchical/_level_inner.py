"""The inner solves of a nested fit on a level system: each member's own step, model and response."""

from __future__ import annotations

import dataclasses
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.optimization.inference.hierarchical._level_solve import (
    _ancestor_values,
    _fold,
    _lower_solve,
    _node_sum,
    _scaled,
    _secular_search,
    _split,
    _upper_solve,
    scaled_system,
)
from diff_soup.optimization.inference.hierarchical._level_system import (
    LevelSystem,
    _addition,
    _dtype,
    _offsets,
    node_of_columns,
)


@functools.partial(jax.jit, static_argnames=("level",))
def _inner_solve(system, damping, *, level):
    """Every member of ``level`` solves its own subtree with its own damping.

    The member's subtree (its block and every level below it) is eliminated
    from the experiments up, as ``_fold`` does, but nothing is folded above
    ``level``: the coordinates there are held. Member ``n`` solves
    ``(H_n + damping_n I) s_n = -g_n``, ``H_n`` and ``g_n`` its subtree's
    blocks, and ``q_n^2 = s_n^T (H_n + damping_n I)^-1 s_n`` comes from the same
    factors. Returns the steps in joint columns (zero above ``level``) and
    ``q^2`` per member.
    """
    counts, widths = system.counts, system.widths
    depth = system.depth
    dtype = system.root.dtype
    offsets = _offsets(counts, widths)

    def own_damping(at: int) -> Any:
        return damping if at == level else damping[system.ancestors[at][level]]

    diagonal = list(system.diagonal)
    cross = [None] + [list(row) for row in system.cross[1:]]
    rhs: list = [None] * (depth + 1)
    for at in range(level, depth + 1):
        rhs[at] = -system.gradient[at]
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
        solved[at] = _lower_solve(factor, rhs[at][..., None])[..., 0]
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
            folded = jnp.einsum("nia,ni->na", row[above], solved[at])
            rhs[above] = rhs[above] - _node_sum(system, folded, at, above)
    steps: list = [None] * (depth + 1)
    if widths[level]:
        factor = jax.vmap(jnp.linalg.cholesky)(
            diagonal[level] + damping[:, None, None] * jnp.eye(widths[level], dtype=dtype)
        )
        factors[level] = factor
        solved[level] = _lower_solve(factor, rhs[level][..., None])[..., 0]
        steps[level] = _upper_solve(factor, solved[level][..., None])[..., 0]
    else:
        steps[level] = jnp.zeros((counts[level - 1], 0), dtype=dtype)
    for at in range(level + 1, depth + 1):
        if not widths[at]:
            steps[at] = jnp.zeros((counts[at - 1], 0), dtype=dtype)
            continue
        right = solved[at]
        for above in range(level, at):
            if not widths[above]:
                continue
            above_steps = _ancestor_values(system, steps[above], at, above)
            right = right - jnp.einsum("nia,na->ni", couplings[at][above], above_steps)
        steps[at] = _upper_solve(factors[at], right[..., None])[..., 0]
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


@functools.partial(jax.jit, static_argnames=("level",))
def _response(system, scale, damping, outer_step, addition, *, level):
    # In the coordinates z = scale * s, with ``addition`` on the scaled diagonal: the
    # levels level .. D are eliminated at the damping with the gradient as right-hand
    # side, then solved from ``level`` down with the outer steps in place of the
    # ancestors' solved steps (no root solve). Also the inner gradient of the scaled
    # model with the outer step held, g_z,i + W_z^T z_o, in joint columns and per
    # inner level.
    scaled = _scaled(system, scale, addition)
    counts, widths = scaled.counts, scaled.widths
    depth = scaled.depth
    dtype = scaled.root.dtype
    offsets = _offsets(counts, widths)
    prefix = offsets[level]
    rhs = [-scaled.root_gradient[:, None]] + [-entry[..., None] for entry in scaled.gradient[1:]]
    _, _, _, _, _, factors, couplings, solved = _fold(scaled, damping, rhs, level)
    outer_z = scale[:prefix] * outer_step
    parts = _split(scaled, jnp.concatenate([outer_z, jnp.zeros((offsets[-1] - prefix,), dtype)]))
    steps: list = [part[..., None] for part in parts[:level]] + [None] * (depth + 1 - level)
    for at in range(level, depth + 1):
        count, width = counts[at - 1], widths[at]
        if width == 0:
            steps[at] = jnp.zeros((count, 0, 1), dtype=dtype)
            continue
        right = solved[at]
        for above in range(at):
            if not widths[above]:
                continue
            if above == 0:
                right = right - jnp.einsum("nia,ak->nik", couplings[at][0], steps[0])
            else:
                above_steps = _ancestor_values(scaled, steps[above], at, above)
                right = right - jnp.einsum("nia,nak->nik", couplings[at][above], above_steps)
        steps[at] = _upper_solve(factors[at], right)
    inner = jnp.concatenate([steps[at].reshape(-1) for at in range(level, depth + 1)])
    shifted = []
    levels = []
    for at in range(level, depth + 1):
        right = scaled.gradient[at]
        for above in range(level):
            if not widths[above] or not widths[at]:
                continue
            if above == 0:
                right = right + jnp.einsum("nia,a->ni", scaled.cross[at][0], parts[0])
            else:
                above_parts = _ancestor_values(scaled, parts[above], at, above)
                right = right + jnp.einsum("nia,na->ni", scaled.cross[at][above], above_parts)
        shifted.append(right.reshape(-1))
        levels.append(right)
    outer_zeros = jnp.zeros((prefix,), dtype)
    return (
        jnp.concatenate([outer_step, inner / scale[prefix:]]),
        jnp.concatenate([outer_zeros, inner]),
        jnp.concatenate([outer_zeros] + shifted),
        tuple(levels),
    )


def inner_solve_scaled(
    system: LevelSystem, scale: Any, level: int, *, addition: Any = None
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

    Returns:
        ``(z, q_squared)`` as host arrays: the scaled steps ``(n,)`` in joint
        columns (zero above ``level``) and ``z_n^T H_n^-1 z_n`` per member
        ``(N_level,)``, ``H_n`` the member's scaled subtree matrix. A member
        whose matrix has no finite factor gets non-finite values.
    """
    dtype = _dtype(system)
    scaled = scaled_system(system, scale, addition)
    count = system.counts[int(level) - 1]
    step, q_squared = _inner_solve(scaled, np.zeros((count,), dtype=dtype), level=int(level))
    return np.asarray(step), np.asarray(q_squared)


def _lockstep_search(
    scaled: LevelSystem, level: int, radii: Any, active: Any, first: Any, *, hold: bool = False
) -> tuple:
    """The multiplier searches of the active members of ``level`` on a scaled system.

    Every active member runs ``_secular_search`` with its own radius on its
    subtree of ``scaled``, the coordinates above ``level`` held. The members
    search in lockstep: every round is one damped solve of all subtrees at
    once, each with its own damping. ``first`` is an optional
    ``(z, q_squared)`` of the solve at zero damping, already made; it answers
    every search's first request and counts as one solve. With ``hold``, a
    member whose search finds no finite step inside its radius takes a zero
    step and a ``nan`` multiplier instead of raising.

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
        step, q_squared = _inner_solve(scaled, damping, level=level)
        solves += 1
        requests = answer(requests, np.asarray(step), np.asarray(q_squared))
    return multipliers, scaled_step, solves


def inner_secular_step(
    system: LevelSystem,
    scale: Any,
    radii: Any,
    level: int,
    active: Any,
    *,
    first: Any = None,
    addition: Any = None,
    hold: bool = False,
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
        scaled, level, radii, active, first, hold=hold
    )
    with np.errstate(all="ignore"):
        delta = scaled_step / scale
    predicted = inner_predicted(system, delta, level)
    if addition is not None:
        predicted -= 0.5 * inner_added(system, addition, scaled_step, level)
    predicted[~active | np.isnan(multipliers)] = 0.0
    return delta, predicted, multipliers, solves


def inner_predicted(system: LevelSystem, delta: Any, level: int) -> np.ndarray:
    """``-(g_n . delta_n + delta_n^T H_n delta_n / 2)`` for every member ``n`` of ``level``.

    ``H_n`` and ``g_n`` are the blocks and the gradient of the member's subtree;
    the columns of ``delta`` above ``level`` do not enter.
    """
    delta = np.asarray(delta, dtype=_dtype(system))
    linear, quadratic = _node_model(system, delta, level=int(level))
    return -(np.asarray(linear, dtype=np.float64) + 0.5 * np.asarray(quadratic, dtype=np.float64))


def model_split(system: LevelSystem, delta: Any, level: int) -> tuple:
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


def inner_quadratic(system: LevelSystem, delta: Any, level: int) -> np.ndarray:
    """``delta_n^T H_n delta_n`` for every member ``n`` of ``level``, ``(N_level,)``.

    ``H_n`` is the matrix of the member's subtree; the columns of ``delta``
    above ``level`` do not enter.
    """
    delta = np.asarray(delta, dtype=_dtype(system))
    _, quadratic = _node_model(system, delta, level=int(level))
    return np.asarray(quadratic, dtype=np.float64)


def inner_added(system: LevelSystem, addition: Any, scaled_step: Any, level: int) -> np.ndarray:
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


def inner_singular(system: LevelSystem, level: int) -> np.ndarray:
    """``(N_level,)``: the members whose subtree's Gauss-Newton matrix has no finite factor."""
    level = int(level)
    count = system.counts[level - 1]
    step, q_squared = _inner_solve(system, np.zeros((count,), dtype=_dtype(system)), level=level)
    owners = node_of_columns(system, level)
    step = np.asarray(step)
    singular = ~np.isfinite(np.asarray(q_squared))
    for member in range(count):
        if not np.all(np.isfinite(step[owners == member])):
            singular[member] = True
    return singular


def inner_programs(system: LevelSystem, level: int) -> None:
    """Compile the programs of the inner problems of ``level`` before a fit's loop.

    These are the members' damped solves, their models, and the solve of the
    members' bounded response (``inner_bounded_response``).
    """
    level = int(level)
    count = system.counts[level - 1]
    dtype = _dtype(system)
    ones = np.ones((system.size,), dtype=dtype)
    scaled = scaled_system(system, ones)
    _inner_solve(scaled, np.ones((count,), dtype=dtype), level=level)
    _node_model(system, np.zeros((system.size,), dtype=dtype), level=level)
    prefix = _offsets(system.counts, system.widths)[level]
    _, _, _, shifted = _response(
        system,
        ones,
        np.asarray(0.0, dtype=dtype),
        np.zeros((prefix,), dtype=dtype),
        np.zeros((system.size,), dtype=dtype),
        level=level,
    )
    _inner_solve(
        _shifted_system(scaled, shifted, level), np.ones((count,), dtype=dtype), level=level
    )


def _response_level(system: LevelSystem, outer_step: np.ndarray) -> int:
    """The cut whose outer columns ``outer_step`` covers; refused otherwise."""
    offsets = _offsets(system.counts, system.widths)
    levels = [
        level
        for level in range(1, system.depth + 1)
        if outer_step.ndim == 1 and offsets[level] == outer_step.shape[0]
    ]
    if not levels:
        raise ValueError(
            "inner_response takes an outer step over the columns above a level of the system, "
            f"of length {', '.join(str(offsets[level]) for level in range(1, system.depth + 1))}; "
            f"got shape {outer_step.shape}."
        )
    return levels[0]


def inner_response_parts(
    system: LevelSystem, scale: Any, damping: Any, outer_step: Any, *, addition: Any = None
) -> tuple:
    """``inner_response`` with the scaled inner step and the inner gradient it answers.

    Args:
        system: The system at the current point, every level.
        scale: ``(n,)``, positive.
        damping: The nonnegative damping of the outer step.
        outer_step: ``(p,)``, the outer step in joint columns.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix (``scaled_system``).

    Returns:
        ``(step, z, shifted)`` as host arrays ``(n,)``: the step of
        ``inner_response``; its inner part in the coordinates
        ``z = scale * s`` (zero on the outer columns); and the inner gradient
        of the scaled model with the outer step held, ``g_{z,i} + W_z^T z_o``
        (zero on the outer columns).

    Raises:
        ValueError: If the length of ``outer_step`` is not the number of columns
            above a level of the system.
    """
    dtype = _dtype(system)
    outer_step = np.asarray(outer_step, dtype=dtype)
    step, z, shifted, _ = _response(
        system,
        np.asarray(scale, dtype=dtype),
        np.asarray(damping, dtype=dtype),
        outer_step,
        _addition(system, addition),
        level=_response_level(system, outer_step),
    )
    return np.asarray(step), np.asarray(z), np.asarray(shifted)


def _shifted_system(scaled: LevelSystem, shifted: tuple, level: int) -> LevelSystem:
    """``scaled`` with the gradients of the levels ``level .. D`` replaced by ``shifted``."""
    return dataclasses.replace(scaled, gradient=tuple(scaled.gradient[:level]) + tuple(shifted))


def inner_bounded_response(
    system: LevelSystem,
    scale: Any,
    damping: Any,
    outer_step: Any,
    radii: Any,
    *,
    addition: Any = None,
) -> tuple:
    """``inner_response_parts`` with every member's response bounded by its radius.

    The members are the nodes of the level whose columns follow those of
    ``outer_step``. A member whose response has a scaled norm ``||z_n||``
    above its radius ``r_n`` responds instead with its own trust-region step
    of radius ``r_n`` at the gradient the response answers: ``z_n`` solves
    ``(V_{z,n} + A_n + lambda_n I) z_n = -(g_{z,n} + W_{z,n}^T z_o)`` over the
    member's subtree, with the multiplier ``lambda_n`` at least ``damping``,
    found by the multiplier search of ``inner_secular_step`` so that
    ``||z_n||`` is ``r_n`` to within ``sqrt(eps) max(1, r_n)`` below it. A
    member within its radius responds as without radii, and a member of
    radius 0 with a nonzero response responds with zero. Fixed inner
    coordinates and held nodes respond with zero, the outer part is
    ``outer_step``.

    Args:
        system: The system at the current point, every level.
        scale: ``(n,)``, positive.
        damping: The nonnegative damping of the outer step.
        outer_step: ``(p,)``, the outer step in joint columns, ``p`` the number
            of columns above a level.
        radii: ``(N_level,)``, every member's radius in the scaled norm, at
            least 0.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix (``scaled_system``).

    Returns:
        ``(step, z, shifted, norms, capped, solves)``: ``step``, ``z`` and
        ``shifted`` as ``inner_response_parts`` returns them, with the parts of
        the bounded members recomputed; ``norms`` ``(N_level,)``, every
        member's scaled response norm before the bound; ``capped``
        ``(N_level,)`` bool, the members whose response was above its radius
        (a response that is not finite counts as above); and the number of
        damped solves the recomputation made.

    Raises:
        ValueError: If the length of ``outer_step`` is not the number of columns
            above a level of the system, if ``radii`` does not hold one finite
            radius of at least 0 per member of that level, or if no finite step
            within its radius is found for a member.
    """
    dtype = _dtype(system)
    outer_step = np.asarray(outer_step, dtype=dtype)
    level = _response_level(system, outer_step)
    count = system.counts[level - 1]
    radii = np.asarray(radii, dtype=np.float64)
    if radii.shape != (count,) or not np.all(np.isfinite(radii) & (radii >= 0.0)):
        raise ValueError(
            "inner_bounded_response takes one finite radius of at least 0 per member of level "
            f"{level}, {count} in all; got {radii!r}."
        )
    scale = np.asarray(scale, dtype=dtype)
    added = _addition(system, addition)
    multiplier = np.asarray(damping, dtype=dtype)
    step, z, shifted, levels = _response(system, scale, multiplier, outer_step, added, level=level)
    step, z, shifted = np.array(step), np.array(z), np.asarray(shifted)
    owners = node_of_columns(system, level)
    below = owners >= 0
    with np.errstate(all="ignore"):
        norms = np.sqrt(np.bincount(owners[below], weights=z[below] ** 2, minlength=count))
    capped = ~(norms <= radii)
    solves = 0
    if capped.any():
        # The bounded members' searches on their subtrees, the outer step's damping on
        # the inner diagonal and the gradient the response answers in place of theirs.
        prefix = _offsets(system.counts, system.widths)[level]
        inner_added = np.array(added, copy=True)
        inner_added[prefix:] += multiplier
        scaled = _shifted_system(_scaled(system, scale, inner_added), levels, level)
        searched = capped & (radii > 0.0)
        scaled_step = np.zeros_like(z)
        if searched.any():
            _, scaled_step, solves = _lockstep_search(scaled, level, radii, searched, None)
        columns = below & capped[np.maximum(owners, 0)]
        z[columns] = scaled_step[columns]
        with np.errstate(all="ignore"):
            step[columns] = scaled_step[columns] / scale[columns]
    return step, z, shifted, norms, capped, solves


def inner_response(
    system: LevelSystem,
    scale: Any,
    damping: Any,
    outer_step: Any,
    *,
    addition: Any = None,
    radii: Any = None,
) -> np.ndarray:
    """The step whose outer part is ``outer_step`` and whose inner part responds to it.

    The outer step covers the columns above a level of the system; that level
    and the levels below it are the inner ones. In the coordinates
    ``z = scale * s`` the inner part minimizes the damped quadratic model with
    the outer part held:
    ``z_i = -(V_z + A + damping I)^-1 (g_{z,i} + W_z^T z_o)``, with ``V_z`` the
    inner blocks, ``A`` the diagonal ``addition`` (zero without one), ``W_z``
    their couplings to the outer blocks and ``g_{z,i}`` the inner gradient of
    the scaled system. It is computed along the tree: the inner levels are
    eliminated from the experiments up at that damping, then solved from the
    cut down with the outer step in place of the solved ancestors. Without an
    addition and at ``damping = 0`` it is the inner part of the joint model's
    minimizer at the fixed outer step, ``-V^-1 (g_i + W^T s_o)``, whatever the
    scale. Fixed inner coordinates and held nodes respond with zero.

    Args:
        system: The system at the current point, every level.
        scale: ``(n,)``, positive.
        damping: The nonnegative damping of the outer step.
        outer_step: ``(p,)``, the outer step in joint columns, ``p`` the number
            of columns above a level.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix (``scaled_system``).
        radii: Optional ``(N_level,)``, one radius per member of the inner
            level, each at least 0: a member whose response has a scaled norm
            above its radius responds with its own step of that radius
            instead (``inner_bounded_response``). ``None`` bounds no member.

    Returns:
        ``(n,)`` as a host array: ``outer_step`` followed by the inner response.

    Raises:
        ValueError: If the length of ``outer_step`` is not the number of columns
            above a level of the system, or ``radii`` does not hold one finite
            radius of at least 0 per member of the inner level.
    """
    if radii is not None:
        bounded = inner_bounded_response(
            system, scale, damping, outer_step, radii, addition=addition
        )
        return bounded[0]
    step, _, _ = inner_response_parts(system, scale, damping, outer_step, addition=addition)
    return step


__all__ = [
    "inner_added",
    "inner_bounded_response",
    "inner_predicted",
    "inner_programs",
    "inner_quadratic",
    "inner_response",
    "inner_response_parts",
    "inner_secular_step",
    "inner_singular",
    "inner_solve_scaled",
]
