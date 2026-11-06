"""Lane selects: a select whose unbatched form is its taken operand, which under vmap picks
each lane's side at every level of nesting and whose tangent and transpose follow the same
per-lane choice."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.utils.lanes import lane_where

_PRED = jnp.asarray([True, False, True, False])
_TAKEN = jnp.arange(12.0).reshape(4, 3)
_OTHER = -1.0 - jnp.arange(12.0).reshape(4, 3)


def _per_lane_where(pred, taken, other):
    """The reference select, ``where`` on each lane."""
    return jnp.where(pred, taken, other)


def test_unbatched_lane_where_returns_the_taken_operand_whatever_the_predicate():
    """Unbatched, the select is its taken operand: a false predicate does not pick the other
    operand, under jit as eagerly."""
    for select in (lane_where, jax.jit(lane_where)):
        np.testing.assert_array_equal(select(False, _TAKEN[0], _OTHER[0]), _TAKEN[0])


def test_lane_where_under_vmap_selects_each_lane_at_every_nesting():
    """Under vmap over a per-lane predicate each lane keeps its own side, with an unbatched
    operand broadcast to the lanes, and an outer vmap's per-lane predicate still selects
    through an inner vmap that batches only the operands."""
    single = jax.vmap(lane_where, in_axes=(0, 0, None))(_PRED, _TAKEN, _OTHER[0])
    np.testing.assert_array_equal(single, jnp.where(_PRED[:, None], _TAKEN, _OTHER[0]))
    inner = jax.vmap(lane_where, in_axes=(None, 0, 0))
    nested = jax.vmap(inner)(_PRED, _TAKEN[:, :, None], _OTHER[:, :, None])
    np.testing.assert_array_equal(nested[..., 0], jnp.where(_PRED[:, None], _TAKEN, _OTHER))


def test_lane_where_tangent_and_transpose_follow_each_lane_choice():
    """Under vmap the forward tangent and the reverse cotangents of the select equal those of
    a per-lane ``where``: each lane's tangent comes from, and its cotangent goes to, the
    operand that lane took."""
    tangents = (jnp.ones_like(_TAKEN), 2.0 * jnp.ones_like(_OTHER))

    def jvp(select):
        return jax.vmap(
            lambda p, a, b, da, db: jax.jvp(partial_select(select, p), (a, b), (da, db))
        )

    def vjp(select):
        def pullback(p, a, b, ct):
            return jax.vjp(partial_select(select, p), a, b)[1](ct)

        return jax.vmap(pullback)

    args = (_PRED, _TAKEN, _OTHER)
    for derived, reference in (
        (jvp(lane_where)(*args, *tangents), jvp(_per_lane_where)(*args, *tangents)),
        (vjp(lane_where)(*args, _TAKEN + 0.5), vjp(_per_lane_where)(*args, _TAKEN + 0.5)),
    ):
        for got, want in zip(derived, reference):
            np.testing.assert_array_equal(got, want)


def partial_select(select, pred):
    """``select`` with its predicate bound, as a function of the two operands."""
    return lambda taken, other: select(pred, taken, other)
