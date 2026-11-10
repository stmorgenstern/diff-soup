"""The inner solves of a nested fit on a level system: each member's own step and model."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from numbers import Real

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from diff_soup.optimization.inference.hierarchical._level_solve import (
    _ancestor_values,
    _carried,
    _lower_solve,
    _node_sum,
    _row_block,
    _scaled,
    _search,
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


def _inner_parts(system, damping, level):
    """Every member of ``level`` solves its own subtree with its own damping.

    The member's subtree (its block and every level below it) is eliminated
    from the experiments up, as ``_eliminate`` does, but nothing is folded
    above ``level``: the coordinates there are held. Member ``n`` solves
    ``(H_n + damping_n I) s_n = -g_n``, ``H_n`` and ``g_n`` its subtree's
    blocks, under its active rows at ``level`` (``system.member_rows[level]``,
    their coefficients on its own coordinates alone): they are eliminated with
    the member's block (``_row_block``), as the joint solve eliminates them.
    ``q_n^2 = s_n^T (H_n + damping_n I)^-1 s_n`` comes from the same factors.
    Returns the steps in joint columns (zero above ``level``), ``q^2`` per
    member, and every member's multipliers of its rows ``(N_level, R)``, those
    of ``(H_n + damping_n I) s_n + g_n + C_n^T mu_n = 0`` (zero on an inactive
    row; ``R`` is 0 without rows).
    """
    counts, widths = system.counts, system.widths
    depth = system.depth
    dtype = system.root.dtype
    offsets = _offsets(counts, widths)
    rows = None if system.member_rows is None else system.member_rows[level]

    def own_damping(at: int) -> jax.Array:
        return damping if at == level else damping[system.ancestors[at][level]]

    diagonal = list(system.diagonal)
    cross = [None] + [list(row) for row in system.cross[1:]]
    rhs: list = [None] * (depth + 1)
    for at in range(level, depth + 1):
        rhs[at] = -system.gradient[at][..., None]
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
    multipliers = jnp.zeros((counts[level - 1], 0), dtype=dtype)
    if widths[level]:
        factor = jax.vmap(jnp.linalg.cholesky)(
            diagonal[level] + damping[:, None, None] * jnp.eye(widths[level], dtype=dtype)
        )
        factors[level] = factor
        solved[level] = _lower_solve(factor, rhs[level])
        right = solved[level]
        if rows is not None:
            # Every member's rows eliminated with its block, the values above held:
            # (W^T W) mu_n = W^T y_n - target_n on its active rows, W = L_n^-1 C_n^T.
            block = _row_block(rows, factor, [None] * level)
            target = jnp.asarray(rows.target, dtype=dtype)[..., None]
            held = _upper_solve(block.factor, _carried(block, right, target))
            multipliers = held[..., 0]
            right = right - jnp.einsum("nir,nrk->nik", block.basis, held)
        steps[level] = _upper_solve(factor, right)[..., 0]
    else:
        steps[level] = jnp.zeros((counts[level - 1], 0), dtype=dtype)
        if rows is not None:
            multipliers = jnp.zeros(np.shape(rows.target), dtype=dtype)
    for at in range(level + 1, depth + 1):
        if not widths[at]:
            steps[at] = jnp.zeros((counts[at - 1], 0), dtype=dtype)
            continue
        right = solved[at][..., 0]
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
    return step, q_squared, multipliers


@functools.partial(jax.jit, static_argnames=("level",))
def _inner_solve(system, damping, *, level):
    """``_inner_parts``'s steps and ``q^2``."""
    step, q_squared, _ = _inner_parts(system, damping, level)
    return step, q_squared


@functools.partial(jax.jit, static_argnames=("level",))
def _inner_multipliers(system, damping, *, level):
    """``_inner_parts``'s multipliers."""
    return _inner_parts(system, damping, level)[2]


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


def inner_solve_scaled(
    system: LevelSystem,
    scale: ArrayLike | Sequence[Real],
    level: int,
    *,
    addition: ArrayLike | Sequence | None = None,
) -> tuple:
    """Every member of ``level`` solves its subtree at zero damping, in ``z = scale * s``.

    The coordinates above ``level`` are held. This is the solve the members'
    multiplier searches of ``inner_secular_step`` make first, for every
    member at once. Every member's step keeps its active rows at ``level``
    (``system.member_rows``).

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
    scale = np.asarray(scale, dtype=dtype)
    scaled = scaled_system(system, scale, addition)
    count = system.counts[int(level) - 1]
    step, q_squared = _inner_solve(scaled, np.zeros((count,), dtype=dtype), level=int(level))
    return np.asarray(step), np.asarray(q_squared)


def inner_row_multipliers(
    system: LevelSystem,
    scale: ArrayLike | Sequence[Real],
    damping: ArrayLike,
    level: int,
    *,
    addition: ArrayLike | Sequence | None = None,
) -> np.ndarray:
    """Every member's multipliers of its active rows at ``level`` at its damped solve in ``z = scale * s``.

    The solve is ``inner_solve_scaled``'s with member ``n`` damped by ``damping[n]``; a
    member's row is kept by its model only where its multiplier is at least zero. An
    inactive row's multiplier is zero.

    Returns:
        ``(N_level, R)`` on the host.
    """
    dtype = _dtype(system)
    scale = np.asarray(scale, dtype=dtype)
    scaled = scaled_system(system, scale, addition)
    multipliers = _inner_multipliers(scaled, np.asarray(damping, dtype=dtype), level=int(level))
    return np.asarray(multipliers, dtype=np.float64)


def _member_sums(values, owners, count):
    """``(count,)``: the sums of ``values`` over every member's columns (traced).

    ``owners`` is ``node_of_columns(system, level)``; a column of no member (``-1``) drops.
    """
    owned = owners >= 0
    return jax.ops.segment_sum(jnp.where(owned, values, 0), jnp.where(owned, owners, 0), count)


def _member_norms(values, owners, count):
    """``(count,)``: the norm of every member's part of ``values`` (traced)."""
    return jnp.sqrt(_member_sums(values * values, owners, count))


def _member_predicted(system, delta, scaled, addition, owners, count, *, level):
    """The decrease every member's scaled model predicts for its part of ``delta`` (traced).

    ``-(g_n . delta_n + delta_n^T H_n delta_n / 2 + z_n^T diag(addition_n) z_n / 2)``, ``z``
    the same step in the scaled coordinates (``scaled``).
    """
    linear, quadratic = _node_model(system, delta, level=level)
    added = _member_sums(addition * scaled * scaled, owners, count)
    return -(linear + 0.5 * quadratic) - 0.5 * added


@functools.partial(jax.jit, static_argnames=("level", "eps", "tiny"))
def _inner_step(system, scale, radii, active, addition, owners, first, *, level, eps, tiny):
    """Every active member's trust-region step on its subtree: one program of the members' searches.

    ``owners`` is ``node_of_columns(system, level)``; ``first`` the members' solve at
    zero damping in these coordinates, or ``None`` to make it. Every member runs the
    device search of ``_level_solve._search`` with its own radius, all in lockstep, one
    damped solve of every subtree per round. Returns ``(delta, predicted, multipliers,
    solves, failed)``: a member whose search finds no finite step inside its radius has
    a zero step, a zero predicted decrease and a ``nan`` multiplier, and is ``failed``.
    """
    count = system.counts[level - 1]
    dtype = system.root.dtype
    scaled = _scaled(system, scale, addition)
    owned = owners >= 0
    member = jnp.where(owned, owners, 0)

    def norms(z):
        return _member_norms(z, owners, count)

    def finite(z):
        return _member_sums((~jnp.isfinite(z)).astype(jnp.int32), owners, count) == 0

    def solve(damping):
        z, q_squared = _inner_solve(scaled, damping, level=level)
        return z, q_squared, jnp.zeros((count,), dtype=bool)

    if first is None:
        zero = solve(jnp.zeros((count,), dtype=dtype))
    else:
        zero = (first[0], first[1], jnp.zeros((count,), dtype=bool))
    multipliers, z, solves, failed, _ = _search(
        solve, zero, radii, eps, tiny, norms=norms, finite=finite, member_of=owners, active=active
    )
    failed = failed & active
    z = jnp.where(owned & failed[member], 0.0, z)
    multipliers = jnp.where(failed, jnp.nan, multipliers)
    delta = z / scale
    predicted = _member_predicted(system, delta, z, addition, owners, count, level=level)
    predicted = jnp.where(~active | jnp.isnan(multipliers), 0.0, predicted)
    return delta, predicted, multipliers, jnp.where(jnp.any(active), solves, 0), failed


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
) -> tuple:
    """The trust-region step of every active member of ``level`` over its own subtree.

    Each active member runs the multiplier search of ``level_secular_step``
    with its own radius in the scaled norm ``||scale * s_n||``, the
    coordinates above ``level`` held. The members search in lockstep: every
    round is one damped solve of all subtrees at once, each with its own
    multiplier. One compiled program (``_inner_step``).

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
    first = None if first is None else tuple(jnp.asarray(part, dtype=dtype) for part in first)
    delta, predicted, multipliers, solves, failed = jax.device_get(
        _inner_step(
            system,
            scale,
            np.asarray(radii, dtype=dtype),
            np.asarray(active, dtype=bool),
            _addition(system, addition),
            node_of_columns(system, level),
            first,
            level=level,
            eps=float(np.finfo(dtype).eps),
            tiny=float(np.finfo(float).tiny),
        )
    )
    if not hold and np.any(failed):
        raise ValueError(
            "TrustRegion could not bracket a finite Gauss-Newton trust-region step; improve "
            "parameter scaling."
        )
    return (
        np.array(delta),
        np.asarray(predicted, dtype=np.float64),
        np.asarray(multipliers, dtype=np.float64),
        int(solves),
    )


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
    "inner_added",
    "inner_row_multipliers",
    "inner_predicted",
    "inner_programs",
    "inner_quadratic",
    "inner_secular_step",
    "inner_solve_scaled",
    "model_split",
]
