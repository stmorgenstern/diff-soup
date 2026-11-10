"""A level system solved by elimination up its tree: the fold, the damped and scaled solves, the attempted step."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Sequence
from numbers import Real
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from diff_soup.optimization.inference.hierarchical._level_system import (
    LevelSystem,
    MemberRows,
    _addition,
    _dtype,
    _offsets,
)

# ---------------------------------------------------------------------------
# The elimination (inside compiled programs)
# ---------------------------------------------------------------------------


def _split(system: LevelSystem, vector: ArrayLike) -> list:
    """A joint vector as the root part and one ``(N_l, v_l)`` part per level."""
    offsets = _offsets(system.counts, system.widths)
    parts = [vector[: system.widths[0]]]
    for level in range(1, system.depth + 1):
        parts.append(
            vector[offsets[level] : offsets[level + 1]].reshape(
                system.counts[level - 1], system.widths[level]
            )
        )
    return parts


def _node_sum(system: LevelSystem, values: ArrayLike, level: int, above: int) -> jax.Array:
    """Sum per-node values of ``level`` over the nodes below each node of ``above``."""
    if above == 0:
        return jnp.sum(values, axis=0)
    table = system.descendants[level][above]
    mask = system.present[level][above]
    gathered = values[table]
    keep = jnp.reshape(mask, mask.shape + (1,) * (values.ndim - 1))
    return jnp.sum(jnp.where(keep, gathered, jnp.zeros((), values.dtype)), axis=1)


def _ancestor_values(system: LevelSystem, values: ArrayLike, level: int, above: int) -> ArrayLike:
    """The values of each node's level-``above`` ancestor, one row per node of ``level``."""
    return values[system.ancestors[level][above]]


def _lower_solve(factor: ArrayLike, right: ArrayLike) -> jax.Array:
    """``L^-1 B`` for a batch of lower factors."""
    return jax.vmap(lambda f, b: jax.scipy.linalg.solve_triangular(f, b, lower=True))(factor, right)


def _upper_solve(factor: ArrayLike, right: ArrayLike) -> jax.Array:
    """``L^-T B`` for a batch of lower factors."""
    return jax.vmap(lambda f, b: jax.scipy.linalg.solve_triangular(f, b, lower=True, trans="T"))(
        factor, right
    )


def _factor(
    diagonal: ArrayLike, cross: Sequence, damping: Real | ArrayLike, widths: tuple, level: int
):
    """Every node's factor ``L_n = chol(diag_n + damping I)`` and its couplings ``L_n^-1 cross_n^a``."""
    factor = jax.vmap(jnp.linalg.cholesky)(
        diagonal + damping * jnp.eye(widths[level], dtype=diagonal.dtype)
    )
    couplings = [
        _lower_solve(factor, cross[above]) if widths[above] else None for above in range(level)
    ]
    return factor, couplings


def _row_sizes(system: LevelSystem, rows: MemberRows, level: int, damping) -> jax.Array:
    """Per member's row, ``sum_j a_j^2 / (H_jj + damping)`` over the columns it reads, ``(N_l, R)``.

    The size of the row's pivot among the multipliers, from the matrix's
    diagonal; it stands for a row whose own coefficients vanish.
    """

    def inverse(diagonal: jax.Array) -> jax.Array:
        diagonal = diagonal + damping
        return jnp.where(diagonal > 0.0, 1.0 / jnp.where(diagonal > 0.0, diagonal, 1.0), 0.0)

    own = inverse(jnp.diagonal(system.diagonal[level], axis1=1, axis2=2))
    sizes = jnp.einsum("nir,ni->nr", rows.coefficients**2, own)
    for above, entry in enumerate(rows.above):
        if not system.widths[above]:
            continue
        if above == 0:
            sizes = sizes + jnp.einsum("nir,i->nr", entry**2, inverse(jnp.diagonal(system.root)))
        else:
            diagonal = jnp.diagonal(system.diagonal[above], axis1=1, axis2=2)
            ancestor = _ancestor_values(system, diagonal, level, above)
            sizes = sizes + jnp.einsum("nir,ni->nr", entry**2, inverse(ancestor))
    return sizes


def _block_factor(gram: ArrayLike, on: ArrayLike, sizes: ArrayLike | None = None) -> jax.Array:
    """``chol`` of every member's multipliers' block, a weak block lifted first.

    A block with a pivot below ``sqrt(eps)`` of its row's size (the larger of
    its diagonal entry and ``sizes``) belongs to a member that holds rows
    dependent on its own values, or a row that reads none of them; its active
    rows' diagonal is raised by that much, so its factor stays finite. A
    joint solve refines the lifted answer to the exact one (``_refined``).
    """
    dtype = gram.dtype
    size = jnp.diagonal(gram, axis1=1, axis2=2)
    if sizes is not None:
        size = jnp.maximum(size, sizes)
    lift = jnp.where(on, jnp.sqrt(jnp.finfo(dtype).eps) * size + jnp.finfo(dtype).tiny, 0.0)
    plain = jax.vmap(jnp.linalg.cholesky)(gram)
    pivots = jnp.diagonal(plain, axis1=1, axis2=2)
    weak = jnp.any(on & ~(pivots * pivots >= lift), axis=1)
    raised = gram + jax.vmap(jnp.diag)(lift)
    return jnp.where(weak[:, None, None], jax.vmap(jnp.linalg.cholesky)(raised), plain)


class _RowBlock(NamedTuple):
    """A level's member rows eliminated with its nodes, every node at once.

    With ``L`` a node's factor and ``B`` its rows' coefficients on its own
    coordinates, ``basis`` is ``W = L^-1 B``; the multipliers' block ``W^T W``,
    an inactive row's row and column the identity's, is ``factor factor^T``
    (``_block_factor``); ``coupled[a]`` is ``factor^-1 (W^T Y^a - B_a^T)``, zero
    on an inactive row, for the couplings ``Y^a = L^-1 cross^a`` and the rows'
    coefficients ``B_a`` on the level-``a`` ancestor (``None`` where the
    ancestor has no width or is held).
    """

    basis: jax.Array
    factor: jax.Array
    coupled: list
    active: jax.Array


def _row_block(
    rows: MemberRows, factor: ArrayLike, couplings: Sequence, sizes: ArrayLike | None = None
) -> _RowBlock:
    """``rows`` eliminated with the nodes of their level, whose factors and couplings are given.

    ``sizes`` are the rows' sizes of ``_row_sizes``, or ``None`` to take the
    multipliers' block's diagonal alone.
    """
    on = jnp.asarray(rows.active, dtype=bool)
    basis = _lower_solve(factor, rows.coefficients)
    gram = jnp.einsum("nir,nis->nrs", basis, basis)
    gram = jnp.where(on[:, :, None] & on[:, None, :], gram, 0.0) + jax.vmap(jnp.diag)(
        jnp.where(on, 0.0, 1.0)
    )
    block_factor = _block_factor(gram, on, sizes)
    coupled = []
    for above, coupling in enumerate(couplings):
        if coupling is None:
            coupled.append(None)
            continue
        value = jnp.einsum("nir,nia->nra", basis, coupling) - jnp.swapaxes(rows.above[above], 1, 2)
        coupled.append(_lower_solve(block_factor, jnp.where(on[..., None], value, 0.0)))
    return _RowBlock(basis, block_factor, coupled, on)


def _carried(block: _RowBlock, solved: ArrayLike, targets: ArrayLike) -> jax.Array:
    """``factor^-1 (W^T y - targets)`` for a level's right-hand sides ``y = L^-1 R``, zero on an inactive row."""
    value = jnp.einsum("nir,nik->nrk", block.basis, solved) - targets
    return _lower_solve(block.factor, jnp.where(block.active[..., None], value, 0.0))


class _Elimination(NamedTuple):
    """The factors of one damped elimination, as ``_eliminate`` makes them.

    Per level: ``factors``, ``couplings`` and the members' ``blocks``
    (``_RowBlock`` or ``None``) of the elimination that holds the members'
    rows, and ``root`` its shared factor. ``plain_factors``,
    ``plain_couplings`` and ``plain_root`` are those of the matrix alone,
    ``H + damping I``, whose inverse-norm ``s^T (H + damping I)^-1 s`` the
    search reads; they differ only above the deepest level with rows.
    """

    factors: list
    couplings: list
    blocks: list
    root: jax.Array | None
    plain_factors: list
    plain_couplings: list
    plain_root: jax.Array | None


def _subtract(system: LevelSystem, blocks: list, level: int, above: int, other: int, product):
    """Subtract ``product``, summed over the nodes of ``level`` below each node of ``above``, from block ``(above, other)``.

    ``blocks`` is ``[root, diagonal, cross]``, changed in place.
    """
    total = _node_sum(system, product, level, above)
    if above == 0:
        blocks[0] = blocks[0] - total
    elif other == above:
        blocks[1][above] = blocks[1][above] - total
    else:
        blocks[2][above][other] = blocks[2][above][other] - total


def _eliminate(system: LevelSystem, damping: Real | ArrayLike) -> _Elimination:
    """Factor the levels ``D .. 1`` from the experiments up, then the shared block.

    Every node of a level is factored at once, ``L_n = chol(diag~_n + damping
    I)``, with its couplings ``Y_n^a = L_n^-1 cross~_n^a``; ``(Y_n^a)^T Y_n^b``
    is folded into the node's ancestors, summed over the nodes below each
    ancestor in node order. A level with members' rows eliminates them with its
    nodes (``_row_block``): each member's multipliers are solved in its own
    block, and its ancestors take ``(Y^a)^T Y^b - coupled[a]^T coupled[b]``,
    the Schur complement of the member's step and multipliers together. The
    work grows with the members, not with the square of the rows.
    """
    widths = system.widths
    depth = system.depth
    dtype = system.root.dtype
    rows = system.member_rows if system.member_rows is not None else (None,) * (depth + 1)
    deepest = max(
        (level for level in range(1, depth + 1) if rows[level] is not None and widths[level]),
        default=0,
    )
    held = [system.root, list(system.diagonal), [None] + [list(row) for row in system.cross[1:]]]
    plain = [system.root, list(system.diagonal), [None] + [list(row) for row in system.cross[1:]]]
    factors: list = [None] * (depth + 1)
    couplings: list = [None] * (depth + 1)
    blocks: list = [None] * (depth + 1)
    plain_factors: list = [None] * (depth + 1)
    plain_couplings: list = [None] * (depth + 1)
    for level in range(depth, 0, -1):
        if not widths[level]:
            continue
        factor, row = _factor(held[1][level], held[2][level], damping, widths, level)
        block = None
        if rows[level] is not None:
            sizes = _row_sizes(system, rows[level], level, damping)
            block = _row_block(rows[level], factor, row, sizes)
        if level < deepest:
            plain_factor, plain_row = _factor(
                plain[1][level], plain[2][level], damping, widths, level
            )
        else:
            plain_factor, plain_row = factor, row
        factors[level], couplings[level], blocks[level] = factor, row, block
        plain_factors[level], plain_couplings[level] = plain_factor, plain_row
        for above in range(level):
            if not widths[above]:
                continue
            for other in range(above + 1):
                if not widths[other]:
                    continue
                product = jnp.einsum("nia,nib->nab", row[above], row[other])
                folded = product
                if block is not None:
                    folded = product - jnp.einsum(
                        "nra,nrb->nab", block.coupled[above], block.coupled[other]
                    )
                _subtract(system, held, level, above, other, folded)
                if above < deepest:
                    if level < deepest:
                        product = jnp.einsum("nia,nib->nab", plain_row[above], plain_row[other])
                    _subtract(system, plain, level, above, other, product)
    g = widths[0]
    root = plain_root = None
    if g:
        identity = damping * jnp.eye(g, dtype=dtype)
        root = jnp.linalg.cholesky(held[0] + identity)
        plain_root = jnp.linalg.cholesky(plain[0] + identity) if deepest else root
    return _Elimination(
        factors, couplings, blocks, root, plain_factors, plain_couplings, plain_root
    )


def _substitute(system: LevelSystem, elimination: _Elimination, rhs: list, targets: list) -> tuple:
    """Solve with ``elimination`` for the right-hand sides ``rhs`` and the members' row ``targets``.

    ``rhs`` holds the root's ``(g, k)`` and every level's ``(N_l, v_l, k)``;
    ``targets[l]`` is ``(N_l, R_l, k)`` at a level with rows. The right-hand
    sides are folded up the tree as the matrix was, the shared block is solved,
    and the steps follow top-down, each member's multipliers with its step.
    Returns ``(steps, multipliers)``: per level the steps ``(N_l, v_l, k)``
    (the root's ``(g, k)`` first), and the multipliers ``(N_l, R_l, k)`` at a
    level with rows.
    """
    counts, widths = system.counts, system.widths
    depth = system.depth
    dtype = system.root.dtype
    rhs = list(rhs)
    solved: list = [None] * (depth + 1)
    carried: list = [None] * (depth + 1)
    for level in range(depth, 0, -1):
        if not widths[level]:
            continue
        row, block = elimination.couplings[level], elimination.blocks[level]
        solved[level] = _lower_solve(elimination.factors[level], rhs[level])
        if block is not None:
            carried[level] = _carried(block, solved[level], targets[level])
        for above in range(level):
            if not widths[above]:
                continue
            folded = jnp.einsum("nia,nik->nak", row[above], solved[level])
            if block is not None:
                folded = folded - jnp.einsum("nra,nrk->nak", block.coupled[above], carried[level])
            rhs[above] = rhs[above] - _node_sum(system, folded, level, above)
    columns = rhs[0].shape[-1]
    if widths[0]:
        half = jax.scipy.linalg.solve_triangular(elimination.root, rhs[0], lower=True)
        top = jax.scipy.linalg.solve_triangular(elimination.root, half, lower=True, trans="T")
    else:
        top = jnp.zeros((0, columns), dtype=dtype)
    steps = [top]
    multipliers: list = [None] * (depth + 1)
    for level in range(1, depth + 1):
        count, width = counts[level - 1], widths[level]
        if width == 0:
            steps.append(jnp.zeros((count, 0, columns), dtype=dtype))
            continue
        row, block = elimination.couplings[level], elimination.blocks[level]
        right, held = solved[level], carried[level]
        for above in range(level):
            if not widths[above]:
                continue
            if above == 0:
                right = right - jnp.einsum("nia,ak->nik", row[0], steps[0])
                if block is not None:
                    held = held - jnp.einsum("nra,ak->nrk", block.coupled[0], steps[0])
            else:
                above_steps = _ancestor_values(system, steps[above], level, above)
                right = right - jnp.einsum("nia,nak->nik", row[above], above_steps)
                if block is not None:
                    held = held - jnp.einsum("nra,nak->nrk", block.coupled[above], above_steps)
        if block is not None:
            multipliers[level] = _upper_solve(block.factor, held)
            right = right - jnp.einsum("nir,nrk->nik", block.basis, multipliers[level])
        steps.append(_upper_solve(elimination.factors[level], right))
    return steps, multipliers


def _stacked(system: LevelSystem, steps: list) -> jax.Array:
    """Per-level steps ``(N_l, v_l, k)`` as joint columns, ``(n, k)``."""
    columns = steps[0].shape[-1]
    return jnp.concatenate(
        [steps[0]] + [steps[level].reshape(-1, columns) for level in range(1, system.depth + 1)],
        axis=0,
    )


def _member_matvec(system: LevelSystem, step_parts: list, multipliers: list) -> tuple:
    """The members' rows times the step, ``(N_l, R_l)``, and their transpose times the multipliers, per level."""
    depth, widths = system.depth, system.widths
    rows = system.member_rows
    times_step: list = [None] * (depth + 1)
    spread = [jnp.zeros_like(part) for part in step_parts]
    for level in range(1, depth + 1):
        if rows[level] is None or not widths[level]:
            continue
        member, mu = rows[level], multipliers[level]
        value = jnp.einsum("nir,ni->nr", member.coefficients, step_parts[level])
        spread[level] = spread[level] + jnp.einsum("nir,nr->ni", member.coefficients, mu)
        for above in range(level):
            if not widths[above]:
                continue
            other = (
                step_parts[0][None, :]
                if above == 0
                else _ancestor_values(system, step_parts[above], level, above)
            )
            value = value + jnp.einsum("nir,ni->nr", member.above[above], other)
            pushed = jnp.einsum("nir,nr->ni", member.above[above], mu)
            spread[above] = spread[above] + _node_sum(system, pushed, level, above)
        times_step[level] = value
    return times_step, spread


def _matvec(system: LevelSystem, parts: list, damping: Real | ArrayLike) -> list:
    """``(H + damping I) s`` from the blocks, per level, for a step given per level."""
    depth, widths = system.depth, system.widths
    out = [system.root @ parts[0] + damping * parts[0]] + [
        jnp.einsum("nij,nj->ni", system.diagonal[level], parts[level]) + damping * parts[level]
        for level in range(1, depth + 1)
    ]
    for level in range(1, depth + 1):
        if not widths[level]:
            continue
        for above in range(level):
            if not widths[above]:
                continue
            coupling = system.cross[level][above]
            if above == 0:
                out[level] = out[level] + jnp.einsum("nij,j->ni", coupling, parts[0])
            else:
                other = _ancestor_values(system, parts[above], level, above)
                out[level] = out[level] + jnp.einsum("nij,nj->ni", coupling, other)
            pushed = jnp.einsum("nij,ni->nj", coupling, parts[level])
            out[above] = out[above] + _node_sum(system, pushed, level, above)
    return out


# The refinements of a joint solve under rows, at most, and the size of a residual it stops at,
# relative to its terms, as a power of eps (ours: a direct solve leaves a few eps; a lifted
# block leaves about sqrt(eps)).
_REFINEMENTS = 8
_REFINED = 0.75


def _refined(system, damping, elimination, answer, general):
    """``answer`` refined until the step and multipliers solve the rows' system to rounding.

    ``answer`` is ``(step, general_multipliers, members)``; ``general`` is
    ``None`` or ``(spread, member_spread, gram)`` of the equality rows, the
    solves of their columns and their masked multipliers' matrix. The residual of ``(H + damping I) s
    + g + A^T mu = 0`` and of every kept row is solved again with the same
    elimination, while it exceeds ``eps^0.75`` of its terms and at most
    ``_REFINEMENTS`` times: a direct solve stops at once, a lifted block
    (``_block_factor``) after a few rounds. A non-finite answer is left as it is.
    Returns ``(step, general_multipliers, members, unrefined)``: ``unrefined`` is true
    when the rounds ran out before the residual reached its tolerance, so the
    answer is the lifted block's.
    """
    depth = system.depth
    dtype = system.root.dtype
    matrix = system.constraint_matrix
    rows = system.member_rows
    gradient = [system.root_gradient] + list(system.gradient[1:])
    general_active = (
        None if system.constraint_active is None else jnp.asarray(system.constraint_active)
    )
    general_target = system.constraint_rhs - matrix @ system.point
    tolerance = jnp.asarray(jnp.finfo(dtype).eps, dtype) ** _REFINED

    def residual(state):
        step, general_mu, members, _ = state
        parts = _split(system, step)
        member_mu = [None] + list(members)
        times_step, spread = _member_matvec(system, parts, member_mu)
        curvature = _matvec(system, parts, damping)
        pushed = _split(system, matrix.T @ general_mu)
        main = [
            -(gradient[level] + curvature[level] + spread[level] + pushed[level])
            for level in range(depth + 1)
        ]
        size = sum(
            jnp.sum(gradient[level] ** 2)
            + jnp.sum(curvature[level] ** 2)
            + jnp.sum((spread[level] + pushed[level]) ** 2)
            for level in range(depth + 1)
        )
        member_rest: list = [None] * (depth + 1)
        error = sum(jnp.sum(entry**2) for entry in main)
        for level in range(1, depth + 1):
            if times_step[level] is None:
                continue
            on = jnp.asarray(rows[level].active, dtype=bool)
            member_rest[level] = jnp.where(on, rows[level].target - times_step[level], 0.0)
            error = error + jnp.sum(member_rest[level] ** 2)
            size = size + jnp.sum(jnp.where(on, rows[level].target, 0.0) ** 2)
            size = size + jnp.sum(jnp.where(on, times_step[level], 0.0) ** 2)
        general_rest = general_target - matrix @ step
        if general_active is not None:
            general_rest = jnp.where(general_active, general_rest, 0.0)
        error = error + jnp.sum(general_rest**2)
        size = size + jnp.sum(general_target**2)
        return main, member_rest, general_rest, error, size

    def needed(state):
        *_, rounds = state
        _, _, _, error, size = residual(state)
        return (rounds < _REFINEMENTS) & jnp.isfinite(error) & (error > tolerance**2 * size)

    def round_(state):
        step, general_mu, members, rounds = state
        main, member_rest, general_rest, _, _ = residual(state)
        rhs = [main[0][:, None]] + [entry[..., None] for entry in main[1:]]
        member_targets = [None] + [
            None if entry is None else entry[..., None] for entry in member_rest[1:]
        ]
        steps, correction = _substitute(system, elimination, rhs, member_targets)
        change = _stacked(system, steps)[:, 0]
        member_change = [None if entry is None else entry[..., 0] for entry in correction[1:]]
        general_change = jnp.zeros_like(general_mu)
        if general is not None:
            spread, member_spread, gram = general
            moved = matrix @ change - general_rest
            if general_active is not None:
                moved = jnp.where(general_active, moved, 0.0)
            general_change = jnp.linalg.solve(gram, moved)
            change = change - spread @ general_change
            member_change = [
                None if entry is None else entry - member_spread[level] @ general_change
                for level, entry in enumerate(member_change, start=1)
            ]
        members = tuple(
            None if entry is None else entry + member_change[level - 1]
            for level, entry in enumerate(members, start=1)
        )
        return step + change, general_mu + general_change, members, rounds + 1

    step, general_mu, members = answer
    state = jax.lax.while_loop(needed, round_, (step, general_mu, tuple(members), jnp.asarray(0)))
    _, _, _, error, size = residual(state)
    unrefined = jnp.isfinite(error) & (error > tolerance**2 * size)
    return state[0], state[1], state[2], unrefined


def _inverse_norm(system: LevelSystem, elimination: _Elimination, step: ArrayLike) -> jax.Array:
    """``s^T (H + damping I)^-1 s`` from the plain factors, ``||L^-1 s||^2`` up the tree."""
    widths, depth = system.widths, system.depth
    parts = _split(system, step)
    halves: list = [None] * (depth + 1)
    q_squared = jnp.zeros((), dtype=system.root.dtype)
    for level in range(depth, 0, -1):
        if not widths[level]:
            continue
        right = parts[level]
        for deeper in range(level + 1, depth + 1):
            if not widths[deeper]:
                continue
            carried = jnp.einsum(
                "nia,ni->na", elimination.plain_couplings[deeper][level], halves[deeper]
            )
            right = right - _node_sum(system, carried, deeper, level)
        halves[level] = _lower_solve(elimination.plain_factors[level], right[..., None])[..., 0]
        q_squared = q_squared + jnp.sum(halves[level] * halves[level])
    if widths[0]:
        right = parts[0]
        for deeper in range(1, depth + 1):
            if not widths[deeper]:
                continue
            carried = jnp.einsum(
                "nia,ni->na", elimination.plain_couplings[deeper][0], halves[deeper]
            )
            right = right - _node_sum(system, carried, deeper, 0)
        top_half = jax.scipy.linalg.solve_triangular(elimination.plain_root, right, lower=True)
        q_squared = q_squared + top_half @ top_half
    return q_squared


def _solve_parts(system, damping):
    """``_solve``'s step and ``q_squared``, and the multipliers of every row the step keeps.

    ``_solve_flagged``'s answer without its flag.
    """
    return _solve_flagged(system, damping)[:3]


def _solve_flagged(system, damping):
    """``(step, q_squared, multipliers, unrefined)``: the damped solve, and whether it is unrefined.

    The multipliers ``mu`` are those of ``(H + damping I) s + g + A^T mu = 0``
    with ``A (point + s) = constraint_rhs`` on the equality rows and every
    member's rows at their targets (``member_rows``); an inactive row's is
    zero. The members' rows are eliminated with their nodes (``_eliminate``),
    the equality rows ride as right-hand-side columns and are solved last, in
    one small system. ``mu`` is ``(c + sum_l N_l R_l,)``: the equality rows,
    then the members' rows level by level, member by member. ``unrefined`` is
    true when the refinement of the members' rows ran out of rounds before its
    tolerance (``_refined``); false without members' rows.
    """
    counts, widths = system.counts, system.widths
    depth = system.depth
    g = widths[0]
    dtype = system.root.dtype
    offsets = _offsets(counts, widths)
    matrix = system.constraint_matrix
    n_rows = matrix.shape[0]
    rows = system.member_rows if system.member_rows is not None else (None,) * (depth + 1)
    rhs = [jnp.concatenate([-system.root_gradient[:, None], matrix[:, :g].T], axis=1)]
    for level in range(1, depth + 1):
        count, width = counts[level - 1], widths[level]
        columns = matrix[:, offsets[level] : offsets[level + 1]].T.reshape(count, width, n_rows)
        rhs.append(jnp.concatenate([-system.gradient[level][..., None], columns], axis=2))
    targets: list = [None] * (depth + 1)
    for level in range(1, depth + 1):
        if rows[level] is not None and widths[level]:
            target = jnp.asarray(rows[level].target, dtype=dtype)
            targets[level] = jnp.concatenate(
                [target[..., None], jnp.zeros(target.shape + (n_rows,), dtype=dtype)], axis=2
            )
    elimination = _eliminate(system, damping)
    steps, member_parts = _substitute(system, elimination, rhs, targets)
    stacked = _stacked(system, steps)
    step = stacked[:, 0]
    members = tuple(None if entry is None else entry[..., 0] for entry in member_parts[1:])
    general = None
    if n_rows:
        spread = stacked[:, 1:]
        member_spread = [None] + [
            None if entry is None else entry[..., 1:] for entry in member_parts[1:]
        ]
        target = system.constraint_rhs - matrix @ system.point
        gram, residual = matrix @ spread, matrix @ step - target
        if system.constraint_active is not None:
            # An inactive row takes a zero multiplier: its row and column of the
            # multipliers' system become the identity's.
            on = system.constraint_active
            identity = jnp.diag(jnp.where(on, 0.0, 1.0))
            gram = jnp.where(on[:, None] & on[None, :], gram, 0.0) + identity
            residual = jnp.where(on, residual, 0.0)
        general_mu = jnp.linalg.solve(gram, residual)
        step = step - spread @ general_mu
        members = tuple(
            None if entry is None else entry - member_spread[level] @ general_mu
            for level, entry in enumerate(members, start=1)
        )
        general = (spread, member_spread, gram)
    else:
        general_mu = jnp.zeros((0,), dtype=dtype)
    unrefined = jnp.asarray(False)
    if system.member_rows is not None:
        step, general_mu, members, unrefined = _refined(
            system,
            damping,
            elimination,
            (step, general_mu, members),
            general,
        )
    multipliers = jnp.concatenate(
        [general_mu] + [entry.reshape(-1) for entry in members if entry is not None]
    )
    return step, _inverse_norm(system, elimination, step), multipliers, unrefined


@jax.jit
def _solve(system, damping):
    step, q_squared, _ = _solve_parts(system, damping)
    return step, q_squared


@jax.jit
def _scaled(system, scale, addition):
    # The system in the coordinates z = scale * s, with ``addition`` added to the
    # diagonal of the scaled matrix.
    inverse = 1.0 / scale
    parts = _split(system, inverse)
    added = _split(system, addition)
    shared = parts[0]
    diagonal: list = [None]
    cross: list = [None]
    gradient: list = [None]
    for level in range(1, system.depth + 1):
        own = parts[level]
        diagonal.append(
            system.diagonal[level] * own[:, :, None] * own[:, None, :]
            + jax.vmap(jnp.diag)(added[level])
        )
        gradient.append(system.gradient[level] * own)
        row = []
        for above in range(level):
            if above == 0:
                other = shared[None, None, :]
            else:
                other = _ancestor_values(system, parts[above], level, above)[:, None, :]
            row.append(system.cross[level][above] * own[:, :, None] * other)
        cross.append(tuple(row))
    member_rows = system.member_rows
    if member_rows is not None:
        # A row's coefficient on a column, over that column's scale.
        member_rows = tuple(
            None
            if rows is None
            else rows._replace(
                coefficients=rows.coefficients * parts[level][:, :, None],
                above=tuple(
                    entry
                    * (
                        shared[None, :, None]
                        if above == 0
                        else _ancestor_values(system, parts[above], level, above)[:, :, None]
                    )
                    for above, entry in enumerate(rows.above)
                ),
            )
            for level, rows in enumerate(member_rows)
        )
    return dataclasses.replace(
        system,
        root=system.root * shared[:, None] * shared[None, :] + jnp.diag(added[0]),
        root_gradient=system.root_gradient * shared,
        diagonal=tuple(diagonal),
        cross=tuple(cross),
        gradient=tuple(gradient),
        point=scale * system.point,
        constraint_matrix=system.constraint_matrix * inverse[None, :],
        member_rows=member_rows,
    )


@jax.jit
def _scaled_multipliers(system, scale, addition, damping):
    return _solve_parts(_scaled(system, scale, addition), damping)[2]


@jax.jit
def _quadratic(system, delta):
    parts = _split(system, delta)
    total = parts[0] @ system.root @ parts[0]
    for level in range(1, system.depth + 1):
        own = parts[level]
        total = total + jnp.einsum("ni,nij,nj->", own, system.diagonal[level], own)
        for above in range(level):
            if not system.widths[above]:
                continue
            if above == 0:
                coupled = jnp.einsum("ni,nij,j->", own, system.cross[level][0], parts[0])
            else:
                other = _ancestor_values(system, parts[above], level, above)
                coupled = jnp.einsum("ni,nij,nj->", own, system.cross[level][above], other)
            total = total + 2.0 * coupled
    return total


@jax.jit
def _vectors(system):
    gradient = jnp.concatenate(
        [system.root_gradient] + [entry.reshape(-1) for entry in system.gradient[1:]]
    )
    diagonal = jnp.concatenate(
        [jnp.diagonal(system.root)]
        + [jnp.diagonal(entry, axis1=1, axis2=2).reshape(-1) for entry in system.diagonal[1:]]
    )
    return gradient, diagonal


# The phases of the multiplier search on the device, as ``_secular_search`` runs it.
_AT_ZERO, _PROBE, _BRACKET, _NEWTON, _FINAL, _NEARLY, _FOUND = range(7)
# The search's iteration limits: the bracket's doublings and the Newton iterations.
_BRACKETS = 80
_NEWTONS = 30


def _norm(vector: ArrayLike) -> jax.Array:
    return jnp.sqrt(jnp.dot(vector, vector))


def _all_finite(vector: ArrayLike) -> jax.Array:
    return jnp.all(jnp.isfinite(vector))


def _search(
    solve: Callable[..., tuple],
    zero: tuple,
    radius: ArrayLike,
    eps: float,
    tiny: float,
    *,
    norms: Callable,
    finite: Callable,
    member_of: ArrayLike,
    active: ArrayLike | None = None,
) -> tuple:
    """``_secular_search`` on the device for ``m`` problems at once: every branch and limit of the host generator.

    Problem ``k`` owns the columns where ``member_of`` is ``k`` (``-1``: none);
    ``norms(z)`` and ``finite(z)`` give every problem's norm of ``z`` and whether
    its part is finite, ``(m,)``. ``solve(damping)`` takes ``(m,)`` dampings (1
    for a problem that asks for none) and gives ``(z, q_squared, unrefined)``,
    ``z`` in joint columns and the others ``(m,)``; ``zero`` is its answer at
    zero damping. ``radius`` is ``(m,)``; a problem outside ``active`` makes no
    search and returns a zero step. Every round is one solve of every problem
    that asks for one. Returns ``(multiplier, z, solves, failed, unrefined)``:
    ``(m,)`` multipliers, the steps in joint columns, the solves made (``zero``
    and every round), whether each problem's bracket found no finite step inside
    its radius, and whether the solve that gave each problem's step is unrefined
    (``_solve_flagged``). The joint attempt is one problem owning every column.
    """
    z0, q0, f0 = zero
    dtype = z0.dtype
    radius = jnp.asarray(radius, dtype=dtype)
    count = radius.shape[0]
    member_of = jnp.asarray(member_of)
    columns = jnp.where(member_of >= 0, member_of, count)
    tolerance = np.sqrt(eps) * jnp.maximum(1.0, radius)
    probe = jnp.full((count,), np.sqrt(eps), dtype=dtype)
    one = jnp.ones((count,), dtype=dtype)
    zero_value = jnp.zeros((count,), dtype=dtype)
    tiny = jnp.asarray(tiny, dtype=dtype)
    machine = jnp.asarray(float(np.finfo(float).eps), dtype=dtype)

    def at_zero(state, n, whole):
        found = whole & jnp.isfinite(state["q"]) & (n <= radius)
        factor_bound = ~jnp.isfinite(state["q"])
        return {
            "phase": jnp.where(found, _FOUND, jnp.where(factor_bound, _PROBE, _BRACKET)),
            "multiplier": zero_value,
            "out": state["z"],
            "out_flag": state["f"],
            "lo": zero_value,
            "lower_is_factor_bound": factor_bound,
            "hi": one,
            "need": ~found,
            "damping": jnp.where(factor_bound, probe, one),
        }

    def at_probe(state, n, whole):
        valid = jnp.isfinite(n) & jnp.isfinite(state["q"])
        found = valid & (n <= radius)
        return {
            "phase": jnp.where(found, _FOUND, _BRACKET),
            "multiplier": probe,
            "out": state["z"],
            "out_flag": state["f"],
            "lo": probe,
            "lower_is_factor_bound": ~valid,
            "hi": one,
            "need": ~found,
            "damping": one,
        }

    def at_bracket(state, n, whole):
        valid = jnp.isfinite(n) & jnp.isfinite(state["q"])
        inside = valid & (n <= radius)
        hi = jnp.where(inside, state["hi"], 2.0 * state["hi"] + 1.0)
        brackets = state["brackets"] + 1
        failed = ~inside & (brackets >= _BRACKETS)
        return {
            "phase": jnp.where(inside, _NEWTON, jnp.where(failed, _FOUND, _BRACKET)),
            "failed": failed,
            "hi_step": jnp.where(spread(inside), state["z"], state["hi_step"]),
            "hi_flag": jnp.where(inside, state["f"], state["hi_flag"]),
            "current": jnp.where(inside, state["hi"], state["current"]),
            "lo": jnp.where(inside, state["lo"], state["hi"]),
            "lower_is_factor_bound": jnp.where(inside, state["lower_is_factor_bound"], ~valid),
            "hi": hi,
            "brackets": brackets,
            "need": ~inside & ~failed,
            "damping": hi,
        }

    def at_newton(state, n, whole):
        q = state["q"]
        current, lo, hi = state["current"], state["lo"], state["hi"]
        broken = ~jnp.isfinite(n) | ~jnp.isfinite(q)
        inside = ~broken & (n <= radius)
        close = inside & (radius - n <= tolerance)
        lo = jnp.where(broken | ~inside, current, lo)
        hi = jnp.where(inside, current, hi)
        bound = jnp.where(broken, True, jnp.where(inside, state["lower_is_factor_bound"], False))
        newton = current + (n * n / jnp.maximum(q, tiny)) * ((n - radius) / radius)
        stepped = jnp.where(
            jnp.isfinite(newton) & (lo < newton) & (newton < hi), newton, 0.5 * (lo + hi)
        )
        current = jnp.where(broken, 0.5 * (lo + hi), stepped)
        more = ~close & (state["newtons"] < _NEWTONS - 1)
        return {
            "phase": jnp.where(more, _NEWTON, _FINAL),
            "hi_step": jnp.where(spread(inside), state["z"], state["hi_step"]),
            "hi_flag": jnp.where(inside, state["f"], state["hi_flag"]),
            "current": current,
            "lo": lo,
            "hi": hi,
            "lower_is_factor_bound": bound,
            "newtons": state["newtons"] + 1,
            "need": more,
            "damping": current,
        }

    def at_final(state, n, whole):
        lo, hi = state["lo"], state["hi"]
        nearly = (
            state["lower_is_factor_bound"]
            & (lo > 0.0)
            & ((hi - lo) <= 1e-7 * jnp.maximum(one, jnp.abs(lo)))
        )
        return {
            "phase": jnp.where(nearly, _NEARLY, _FOUND),
            "multiplier": hi,
            "out": state["hi_step"],
            "out_flag": state["hi_flag"],
            "need": nearly,
            "damping": lo + jnp.maximum(1e-6 * jnp.abs(lo), machine),
        }

    def at_nearly(state, n, whole):
        good = whole & jnp.isfinite(state["q"])
        return {
            "phase": jnp.full((count,), _FOUND),
            "multiplier": jnp.where(good, state["damping"], state["multiplier"]),
            "out": jnp.where(spread(good), state["z"], state["out"]),
            "out_flag": jnp.where(good, state["f"], state["out_flag"]),
            "need": jnp.zeros((count,), dtype=bool),
        }

    def spread(flags):
        """Per-problem flags on the columns, a column of no problem false."""
        return jnp.concatenate([flags, jnp.zeros((1,), dtype=bool)])[columns]

    branches = (at_zero, at_probe, at_bracket, at_newton, at_final, at_nearly)
    column_fields = ("z", "out", "hi_step")

    def body(state):
        n, whole = norms(state["z"]), finite(state["z"])
        made = [branch(state, n, whole) for branch in branches]
        phase = state["phase"]
        chosen = dict(state)
        for key in made[0].keys() | {key for update in made for key in update}:
            options = [update.get(key, state[key]) for update in made] + [state[key]]
            stacked = jnp.stack(options)
            if key in column_fields:
                index = jnp.concatenate([phase, jnp.full((1,), _FOUND)])[columns]
            else:
                index = phase
            chosen[key] = jnp.take_along_axis(stacked, index[None], axis=0)[0]
        chosen["phase"] = chosen["phase"].astype(jnp.int32)
        need = chosen["need"]
        z, q, f = jax.lax.cond(
            jnp.any(need),
            lambda: solve(jnp.where(need, chosen["damping"], 1.0)),
            lambda: (chosen["z"], chosen["q"], chosen["f"]),
        )
        return {
            **chosen,
            "z": jnp.where(spread(need), z, chosen["z"]),
            "q": jnp.where(need, q, chosen["q"]),
            "f": jnp.where(need, f, chosen["f"]),
            "solves": chosen["solves"] + jnp.any(need),
        }

    begin = jnp.ones((count,), dtype=bool) if active is None else jnp.asarray(active, dtype=bool)
    zeros = jnp.zeros_like(z0)
    state = {
        "phase": jnp.where(begin, _AT_ZERO, _FOUND).astype(jnp.int32),
        "z": z0,
        "f": jnp.asarray(f0, dtype=bool),
        "hi_flag": jnp.zeros((count,), dtype=bool),
        "out_flag": jnp.zeros((count,), dtype=bool),
        "q": jnp.asarray(q0, dtype=dtype),
        "damping": zero_value,
        "need": jnp.zeros((count,), dtype=bool),
        "lo": zero_value,
        "hi": one,
        "current": one,
        "lower_is_factor_bound": jnp.zeros((count,), dtype=bool),
        "hi_step": zeros,
        "out": zeros,
        "multiplier": zero_value,
        "brackets": jnp.zeros((count,), dtype=jnp.int32),
        "newtons": jnp.zeros((count,), dtype=jnp.int32),
        "solves": jnp.asarray(1, dtype=jnp.int32),
        "failed": jnp.zeros((count,), dtype=bool),
    }
    state = jax.lax.while_loop(lambda state: jnp.any(state["phase"] != _FOUND), body, state)
    return state["multiplier"], state["out"], state["solves"], state["failed"], state["out_flag"]


@functools.partial(jax.jit, static_argnames=("eps", "tiny"))
def _attempt(system, scale, addition, radius, *, eps, tiny):
    # One attempt's linear algebra: the scaled system, its solve at zero damping and the
    # decrease the model predicts for that full step, then the multiplier search of
    # radius ``radius`` and the decrease predicted for its step.
    scaled = _scaled(system, scale, addition)
    gradient = jnp.concatenate(
        [system.root_gradient] + [entry.reshape(-1) for entry in system.gradient[1:]]
    )
    inverse = 1.0 / scale

    def solve(damping):
        step, q_squared, _, unrefined = _solve_flagged(scaled, damping[0])
        return step, q_squared[None], unrefined[None]

    def decrease(z):
        step = inverse * z
        quadratic = _quadratic(system, step) + jnp.sum(
            jnp.where(addition != 0.0, addition * (z * z), 0.0)
        )
        return -(gradient @ step + 0.5 * quadratic)

    zero = solve(jnp.zeros((1,), scale.dtype))
    # A zero-damping solve whose rows' refinement ran out predicts no decrease to test.
    finite = _all_finite(zero[0]) & jnp.isfinite(zero[1][0]) & ~zero[2][0]
    full = jnp.where(finite, decrease(jnp.where(finite, zero[0], 0.0)), jnp.nan)
    multiplier, scaled_step, solves, failed, unrefined = _search(
        solve,
        zero,
        jnp.reshape(radius, (1,)),
        eps,
        tiny,
        norms=lambda z: _norm(z)[None],
        finite=lambda z: _all_finite(z)[None],
        member_of=np.zeros(scale.shape, dtype=np.int32),
    )
    # A step whose solve is unrefined fails as a step the bracket did not find.
    multiplier, failed = multiplier[0], failed[0] | unrefined[0]
    zero = (zero[0], zero[1][0])
    return (
        zero[0],
        zero[1],
        full,
        inverse * scaled_step,
        decrease(scaled_step),
        multiplier,
        solves,
        failed,
    )


def _added_quadratic(addition: ArrayLike | Sequence | None, scaled_step: np.ndarray) -> float:
    """``z^T diag(addition) z`` for a scaled step ``z``; 0 without an addition."""
    if addition is None:
        return 0.0
    with np.errstate(all="ignore"):
        return float(np.asarray(addition, dtype=np.float64) @ (scaled_step * scaled_step))


def scaled_system(
    system: LevelSystem,
    scale: ArrayLike | Sequence[Real],
    addition: ArrayLike | Sequence | None = None,
) -> LevelSystem:
    """The system in the coordinates ``z = scale * s``, ``addition`` added to its diagonal.

    Its matrix is ``D H D + diag(addition)`` and its gradient ``D g``, with
    ``D = diag(1 / scale)``; with a unit scale it is the system with the
    addition on the diagonal of its matrix.

    Args:
        system: The system.
        scale: ``(n,)``, positive.
        addition: Optional ``(n,)``, nonnegative; ``None`` adds nothing.

    Returns:
        A [LevelSystem][diff_soup.optimization.inference.hierarchical._level_system.LevelSystem]
        of the same shape.
    """
    return _scaled(system, np.asarray(scale, dtype=_dtype(system)), _addition(system, addition))


def solve_damped(system: LevelSystem, damping: Real | ArrayLike) -> tuple:
    """Solve ``(H + damping I) s = -g`` along the tree, under the system's equality and members' rows.

    Args:
        system: The system.
        damping: The nonnegative damping ``lambda``.

    Returns:
        ``(step, q_squared)``: the step ``(n,)`` in joint columns and
        ``s^T (H + damping I)^-1 s``. A matrix that is not positive definite at
        this damping gives non-finite values.
    """
    return _solve(system, np.asarray(damping, dtype=_dtype(system)))


def solve_scaled(
    system: LevelSystem,
    scale: ArrayLike | Sequence[Real],
    damping: Real | ArrayLike,
    addition: ArrayLike | Sequence | None = None,
) -> tuple:
    """One damped solve in the coordinates ``z = scale * s``, on the host.

    The matrix is the scaled one with ``addition`` on its diagonal
    (``scaled_system``).

    Returns:
        ``(z, q_squared)`` as a host array and a float.
    """
    scaled = scaled_system(system, scale, addition)
    step, q_squared = _solve(scaled, np.asarray(damping, dtype=_dtype(system)))
    return np.asarray(step), float(q_squared)


def row_multipliers(
    system: LevelSystem,
    scale: ArrayLike | Sequence[Real],
    damping: Real | ArrayLike,
    addition: ArrayLike | Sequence | None = None,
) -> np.ndarray:
    """The multipliers of the system's rows at ``solve_scaled``'s solve, on the host.

    Row ``i``'s multiplier ``mu_i`` is that of ``(H~ + damping I) z + g~ + A~^T mu = 0`` in the
    scaled coordinates, which is that of the unscaled rows: a row held at a side of an
    inequality is kept by the step's model only where its multiplier is at least zero. An
    inactive row's is zero.

    Returns:
        ``(c + sum_l N_l R_l,)``: one per row of ``system.constraint_matrix``, then one per
        row of ``system.member_rows``, level by level, member by member.
    """
    dtype = _dtype(system)
    multipliers = _scaled_multipliers(
        system,
        np.asarray(scale, dtype=dtype),
        _addition(system, addition),
        np.asarray(damping, dtype=dtype),
    )
    return np.asarray(multipliers, dtype=np.float64)


def level_quadratic(system: LevelSystem, delta: ArrayLike | Sequence[Real]) -> float:
    """``delta^T H delta`` from the blocks, without a dense matrix.

    Args:
        system: The system.
        delta: ``(n,)``, a step in joint columns.

    Returns:
        The quadratic form as a float.
    """
    return float(_quadratic(system, np.asarray(delta, dtype=_dtype(system))))


def level_vectors(system: LevelSystem) -> tuple:
    """The gradient and the diagonal of the matrix in joint columns, as host arrays."""
    gradient, diagonal = _vectors(system)
    return np.asarray(gradient), np.asarray(diagonal)


def level_secular_step(
    system: LevelSystem,
    scale: ArrayLike | Sequence[Real],
    radius: Real | ArrayLike,
    *,
    first: tuple[ArrayLike, Real | ArrayLike] | AttemptSolve | None = None,
    addition: ArrayLike | Sequence | None = None,
) -> tuple:
    """The trust-region step of radius ``radius`` in the scaled norm ``||scale * s||``.

    The multiplier ``lambda`` is zero when the Gauss-Newton step fits;
    otherwise it is found by a safeguarded Newton iteration on the secular
    equation ``||z(lambda)|| = radius`` above the Cholesky bound, each trial a
    damped solve along the tree in the scaled coordinates.

    Args:
        system: The system.
        scale: ``(n,)``, positive.
        radius: The trust-region radius.
        first: Optional ``(z, q_squared)`` of the solve at ``lambda = 0`` in
            these coordinates, already made; it counts as this step's first
            solve.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix (``scaled_system``); the model whose step is solved
            and whose decrease is predicted carries it.

    Returns:
        ``(step, predicted, multiplier, solves)``: the step in joint columns,
        the decrease the model predicts, the multiplier, and the number of
        damped solves made.

    Raises:
        ValueError: If no finite step inside the radius is found.
    """
    dtype = _dtype(system)
    scale = np.asarray(scale, dtype=dtype)
    scaled = scaled_system(system, scale, addition)
    inverse_scale = 1.0 / scale
    eps = float(np.finfo(dtype).eps)
    search = _secular_search(float(radius), eps, float(np.finfo(float).tiny))
    gradient, _ = level_vectors(system)
    # Every damping the search asks for is one damped solve; the solve at zero damping
    # given as ``first`` answers its first request.
    damping = next(search)
    answer = None if first is None else (np.asarray(first[0]), float(first[1]))
    solves = 0
    try:
        while True:
            if answer is None:
                z, q_squared = _solve(scaled, np.asarray(damping, dtype=dtype))
                answer = np.asarray(z), float(q_squared)
            solves += 1
            damping, answer = search.send(answer), None
    except StopIteration as found:
        multiplier, scaled_step = found.value
    with np.errstate(all="ignore"):
        step = inverse_scale * scaled_step
    quadratic = level_quadratic(system, step) + _added_quadratic(addition, scaled_step)
    predicted = -(float(gradient @ step) + 0.5 * quadratic)
    return step, predicted, float(multiplier), solves


class AttemptSolve(NamedTuple):
    """The linear algebra of one attempt of the joint fit, from one program on the device.

    Indexed as a pair it is the solve at zero damping ``(z, q_squared)`` that
    ``solve_scaled`` gives, so it stands wherever that pair is taken.

    Attributes:
        z: The full step at zero damping in the coordinates ``z = scale * s``.
        q_squared: Its inverse-norm ``z^T (H_z)^-1 z``.
        decrease: The decrease the model predicts for the full step ``z /
            scale``, in cost units, not divided by anything; ``nan`` when the
            solve at zero damping is not finite.
        step: The step of radius ``radius`` in joint columns, as
            ``level_secular_step`` gives it.
        predicted: The decrease the model predicts for ``step``.
        multiplier: The multiplier of ``step``.
        solves: The damped solves the search made, the one at zero damping
            included.
        radius: The radius ``step`` was solved for.
        scale: The scale of the coordinates.
        addition: The diagonal added to the scaled matrix, or ``None``.
        failed: Whether the search found no finite step inside the radius
            (a point off its equality rows whose restoring step exceeds the
            radius, or a scaling no damping cures). ``z``, ``q_squared`` and
            ``decrease`` stand either way; ``step``, ``predicted`` and
            ``multiplier`` are then zero, and the loop that takes the step
            raises there, as ``level_secular_step`` does, so a fit that ends
            before it takes a step is not ended by the search.
    """

    z: np.ndarray
    q_squared: float
    decrease: float
    step: np.ndarray
    predicted: float
    multiplier: float
    solves: int
    radius: float
    scale: np.ndarray
    addition: np.ndarray | None
    failed: bool = False

    def answers(
        self,
        scale: ArrayLike | Sequence[Real],
        radius: Real | ArrayLike,
        addition: ArrayLike | Sequence | None = None,
    ) -> bool:
        """Whether this attempt was solved for ``scale``, ``radius`` and ``addition``."""
        if float(radius) != self.radius or not np.array_equal(np.asarray(scale), self.scale):
            return False
        if addition is None or self.addition is None:
            return addition is None and self.addition is None
        return np.array_equal(np.asarray(addition), self.addition)


def attempt_solve(
    system: LevelSystem,
    scale: ArrayLike | Sequence[Real],
    radius: Real | ArrayLike,
    addition: ArrayLike | Sequence | None = None,
) -> AttemptSolve:
    """One attempt's solve at zero damping, its predicted decrease and its trust-region step.

    The same numbers as ``solve_scaled(system, scale, 0.0, addition)``, the
    full step's predicted decrease, and ``level_secular_step(system, scale,
    radius, first=..., addition=addition)``, from one compiled program per
    system shape and one transfer to the host: the multiplier search runs on
    the device, branch for branch as on the host. The numbers agree with that
    chain to rounding.

    Args:
        system: The system.
        scale: ``(n,)``, positive.
        radius: The trust-region radius.
        addition: Optional ``(n,)``, nonnegative, added to the diagonal of the
            scaled matrix.

    Returns:
        The [AttemptSolve][diff_soup.optimization.inference.hierarchical._level_solve.AttemptSolve].
        A search that finds no finite step inside the radius sets its
        ``failed``; nothing is raised here, so the solve at zero damping and
        its predicted decrease serve the tests a loop makes before it takes a
        step.
    """
    dtype = _dtype(system)
    scale = np.asarray(scale, dtype=dtype)
    added = _addition(system, addition)
    outputs = _attempt(
        system,
        scale,
        added,
        np.asarray(radius, dtype=dtype),
        eps=float(np.finfo(dtype).eps),
        tiny=float(np.finfo(float).tiny),
    )
    z, q_squared, decrease, step, predicted, multiplier, solves, failed = jax.device_get(outputs)
    return AttemptSolve(
        z=np.asarray(z),
        q_squared=float(q_squared),
        decrease=float(decrease),
        step=np.array(step),
        predicted=float(predicted),
        multiplier=float(multiplier),
        solves=int(solves),
        radius=float(radius),
        scale=scale,
        addition=None if addition is None else added,
        failed=bool(failed),
    )


def _secular_search(radius: float, eps: float, tiny: float):
    """The multiplier search of one trust-region problem of radius ``radius``, as a generator.

    It yields the damping of the next solve and receives ``(z, q_squared)``,
    the problem's scaled step at that damping and its inverse-norm; it returns
    ``(multiplier, z)``. ``level_secular_step`` runs one for the whole system;
    ``_search`` is the same search compiled, for any number of problems at once.
    """

    def norm(vector: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            return float(np.linalg.norm(vector))

    def finite(vector: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(vector)))

    scaled_step, zero_q_squared = yield 0.0
    if finite(scaled_step) and np.isfinite(zero_q_squared) and norm(scaled_step) <= radius:
        return 0.0, scaled_step

    lo = 0.0
    lower_is_factor_bound = not np.isfinite(zero_q_squared)
    if lower_is_factor_bound:
        probe = float(np.sqrt(eps))
        probe_step, probe_q_squared = yield probe
        probe_norm = norm(probe_step)
        probe_valid = np.isfinite(probe_norm) and np.isfinite(probe_q_squared)
        if probe_valid and probe_norm <= radius:
            return probe, probe_step
        lo = probe
        lower_is_factor_bound = not probe_valid
    hi = 1.0
    for _ in range(80):
        candidate, q_squared = yield hi
        candidate_norm = norm(candidate)
        valid_factor = np.isfinite(candidate_norm) and np.isfinite(q_squared)
        if valid_factor and candidate_norm <= radius:
            hi_step = candidate
            break
        lo = hi
        lower_is_factor_bound = not valid_factor
        hi = 2.0 * hi + 1.0
    else:
        raise ValueError(
            "TrustRegion could not bracket a finite Gauss-Newton trust-region step; improve "
            "parameter scaling."
        )

    current = hi
    tolerance = float(np.sqrt(eps)) * max(1.0, radius)
    for iteration in range(30):
        candidate_norm = norm(candidate)
        if not np.isfinite(candidate_norm) or not np.isfinite(q_squared):
            lo = current
            lower_is_factor_bound = True
            current = 0.5 * (lo + hi)
            if iteration < 29:
                candidate, q_squared = yield current
            continue
        elif candidate_norm <= radius:
            hi = current
            hi_step = candidate
            if radius - candidate_norm <= tolerance:
                break
        else:
            lo = current
            lower_is_factor_bound = False
        newton = current + (candidate_norm * candidate_norm / max(q_squared, tiny)) * (
            (candidate_norm - radius) / radius
        )
        current = newton if np.isfinite(newton) and lo < newton < hi else 0.5 * (lo + hi)
        if iteration < 29:
            candidate, q_squared = yield current

    multiplier = hi
    scaled_step = hi_step
    if lower_is_factor_bound and lo > 0.0 and (hi - lo) <= 1e-7 * max(1.0, abs(lo)):
        nearly_exact = lo + max(1e-6 * abs(lo), float(np.finfo(float).eps))
        nearly_exact_step, nearly_exact_q_squared = yield nearly_exact
        if finite(nearly_exact_step) and np.isfinite(nearly_exact_q_squared):
            multiplier = nearly_exact
            scaled_step = nearly_exact_step
    return multiplier, scaled_step


__all__ = [
    "AttemptSolve",
    "attempt_solve",
    "level_quadratic",
    "level_secular_step",
    "level_vectors",
    "row_multipliers",
    "scaled_system",
    "solve_damped",
    "solve_scaled",
]
