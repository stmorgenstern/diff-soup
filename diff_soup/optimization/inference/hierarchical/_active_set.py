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
method on its own, all in lockstep, one ``solve`` per round. The method is one bounded loop of
a compiled program, at a fixed shape: the working set is a mask over every row.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from diff_soup.optimization.solvers._loop_judge import _namespace

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
        solves: The damped solves of every ``solve`` made and one per multiplier solve.
    """

    active: jax.Array
    answer: Any
    step: jax.Array
    settled: jax.Array
    solves: jax.Array


def slack(c: ArrayLike, C: ArrayLike, step: ArrayLike) -> ArrayLike:
    """A row's slack for rounding: a few hundred eps of the sizes its linearization adds (for arrays)."""
    xp = _namespace(c, C, step)
    eps = float(np.finfo(np.float64).eps)
    return 256.0 * eps * (xp.abs(c) + xp.abs(C) @ xp.abs(step)) + float(np.finfo(np.float64).tiny)


def _threshold(dtype: Any) -> float:
    """``_DEPENDENT`` in double precision, scaled with the square root of ``dtype``'s eps."""
    return _DEPENDENT * float(np.sqrt(np.finfo(dtype).eps / np.finfo(np.float64).eps))


def _off_span(C: jax.Array, held: jax.Array, threshold: float) -> jax.Array:
    """``(G, P)``: whether each row of ``C`` keeps a part off the span of its group's ``held`` rows.

    ``C`` is ``(G, P, n)``, every group's rows; ``held`` is ``(G, P)``. A row, normalized,
    is projected onto the span of its group's held rows (at a fixed shape: a row not held
    has the identity's row in the Gram matrix), and is independent when more than
    ``threshold`` of its norm is left. A zero row is not independent.
    """
    norms = jnp.linalg.norm(C, axis=-1)
    vectors = C / jnp.where(norms > 0.0, norms, 1.0)[..., None]
    basis = vectors * held[..., None]
    gram = jnp.einsum("gki,gli->gkl", basis, basis)
    gram = gram + jax.vmap(jnp.diag)(jnp.where(held, 0.0, 1.0).astype(C.dtype))
    weights = jnp.linalg.solve(gram, jnp.einsum("gki,gri->gkr", basis, vectors))
    left = vectors - jnp.einsum("gkr,gki->gri", weights, basis)
    return (norms > 0.0) & (jnp.linalg.norm(left, axis=-1) > threshold)


def _starting(c: jax.Array, C: jax.Array, x: jax.Array, shape: tuple) -> jax.Array:
    """The starting set: the rows at their boundary at ``x``, independent within each group.

    A row is at its boundary within ``sqrt(eps)`` of its size ``|c| + |C| |x|``. A row with
    no column a step moves, or one whose coefficients lie in the span of the group's rows
    taken before it, is left out. ``shape`` is ``(G, P)``; rows are taken in their order
    within a group, one row of every group at a time.
    """
    groups, per = shape
    eps = float(np.finfo(np.float64).eps)
    size = jnp.abs(c) + jnp.abs(C) @ jnp.abs(x)
    near = ((c >= -np.sqrt(eps) * size) & jnp.any(C != 0.0, axis=1)).reshape(groups, per)
    rows = C.reshape(groups, per, -1)
    threshold = _threshold(C.dtype)

    def take(position: Any, chosen: jax.Array) -> jax.Array:
        free = _off_span(rows, chosen, threshold)[:, position]
        return chosen.at[:, position].set(near[:, position] & free)

    return jax.lax.fori_loop(0, per, take, jnp.zeros((groups, per), dtype=bool)).reshape(-1)


def active_set(
    solve: Callable,
    multipliers: Callable,
    c: jax.Array,
    C: jax.Array,
    x: jax.Array,
    first: Any,
    *,
    shape: tuple,
    columns: np.ndarray,
) -> Active:
    """The step under the rows ``c + C s <= 0`` by the primal active-set method, one bounded loop.

    ``c`` ``(R,)`` are the rows at ``x`` and ``C`` ``(R, n)`` their Jacobian, zero on the
    columns a step never moves. The rows are ``shape = (G, P)`` groups of ``P`` rows each,
    group by group; ``columns``
    gives the group of every column (``-1`` for a column of no group), static. ``solve(active)``
    returns an answer whose ``step`` is the step ``(n,)``, whose ``solves`` counts its damped
    solves and whose ``failed`` says that no finite step keeps the set within the radius;
    ``multipliers(active, answer)`` gives the held rows' multipliers ``(R,)``. ``first`` is
    ``solve``'s answer with no row held, already made and not counted, or ``None``: the
    starting set is then solved whether it holds a row or not. The round limit is
    ``4 + 2 P``; a solve that fails ends the method with the set and the answer before it,
    the groups that changed the set in that round unsettled. The independence threshold and
    the multipliers' tolerance follow the dtype of ``C``. ``Active.solves`` counts the damped
    solves of every solve made and one per multiplier solve.
    """
    groups, per = shape
    count = groups * per
    group_of = np.repeat(np.arange(groups), per)
    columns = np.asarray(columns, dtype=np.int64)
    in_group = (columns[None, :] == np.arange(groups)[:, None]).astype(np.dtype(C.dtype))
    threshold = _threshold(C.dtype)
    eps = float(np.finfo(C.dtype).eps)
    active = _starting(c, C, x, shape)
    if first is None:
        answer = solve(active)
        solves = jnp.asarray(answer.solves)
    else:
        answer, solves = jax.lax.cond(
            jnp.any(active),
            lambda: (lambda made: (made, jnp.asarray(made.solves)))(solve(active)),
            lambda: (first, jnp.zeros_like(jnp.asarray(first.solves))),
        )
    rows = C.reshape(groups, per, -1)
    state = (
        jnp.asarray(0),
        jnp.asarray(True),
        active,
        answer,
        jnp.zeros_like(x),
        jnp.zeros((groups,), dtype=bool),
        solves,
    )

    def going(state: tuple) -> jax.Array:
        return (state[0] < _EXTRA_ROUNDS + 2 * per) & state[1]

    def round_(state: tuple) -> tuple:
        rounds, _, active, answer, reached, settled, solves = state
        step = answer.step
        open_rows = ~settled[group_of]
        move = step - reached
        rate = C @ move
        crossed = open_rows & ~active & (c + C @ step > slack(c, C, step))
        fraction = jnp.maximum(-(c + C @ reached), 0.0) / jnp.where(rate > 0.0, rate, jnp.inf)
        fraction = jnp.where(crossed & jnp.isfinite(fraction), fraction, 0.0)
        eligible = crossed & _off_span(rows, active.reshape(groups, per), threshold).reshape(-1)
        key = jnp.where(eligible, fraction, jnp.inf).reshape(groups, per)
        first_row = jnp.argmin(key, axis=1)
        joining = ~settled & jnp.any(eligible.reshape(groups, per), axis=1)
        held = active.reshape(groups, per)
        holding = jnp.any(held, axis=1)
        asked = jnp.any(~settled & ~joining & holding)
        mu = jax.lax.cond(
            asked,
            lambda: jnp.asarray(multipliers(active, answer), dtype=C.dtype),
            lambda: jnp.zeros((count,), dtype=C.dtype),
        ).reshape(groups, per)
        solves = solves + jnp.where(asked, 1, 0)
        valued = held & ~jnp.isnan(mu)
        tolerance = np.sqrt(eps) * jnp.max(jnp.where(valued, jnp.abs(mu), 0.0), axis=1)
        lowest = jnp.where(valued, mu, jnp.inf)
        releasing = ~settled & ~joining & (jnp.min(lowest, axis=1) < -tolerance)
        last_row = jnp.argmin(lowest, axis=1)
        settling = ~settled & ~joining & ~releasing
        rows_index = jnp.arange(per)[None, :]
        join_mask = joining[:, None] & (rows_index == first_row[:, None])
        release_mask = releasing[:, None] & (rows_index == last_row[:, None])
        proposed = ((held | join_mask) & ~release_mask).reshape(-1)
        taken = jnp.sum(jnp.where(join_mask, key, 0.0), axis=1)
        share = jnp.where(joining, jnp.minimum(taken, 1.0), 0.0) @ in_group
        moving = (jnp.where(joining, 0.0, 1.0) * jnp.where(settled, 0.0, 1.0)) @ in_group
        reached = jnp.where(moving > 0.0, step, reached + share * move)
        settled = settled | settling
        changed = jnp.any(joining | releasing)

        def resolve() -> tuple:
            again = solve(proposed)
            failed = jnp.asarray(again.failed)
            return (
                jnp.where(failed, active, proposed),
                jax.tree_util.tree_map(lambda new, old: jnp.where(failed, old, new), again, answer),
                ~failed,
                solves + jnp.where(failed, 0, again.solves),
            )

        active, answer, going_on, solves = jax.lax.cond(
            changed, resolve, lambda: (active, answer, jnp.asarray(False), solves)
        )
        return rounds + 1, going_on, active, answer, reached, settled, solves

    _, _, active, answer, reached, settled, solves = jax.lax.while_loop(going, round_, state)
    return Active(active, answer, reached, settled, solves)


__all__ = ["Active", "active_set", "slack"]
