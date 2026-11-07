"""The lane conditional: one conditional unbatched whose lanes under vmap each keep their own
branch at every level of nesting, with derivatives that follow the same choice."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.utils import lane_cond, live_lane_cond

_PRED = jnp.asarray([True, False, True, False])
_TAKEN = jnp.arange(12.0).reshape(4, 3)
_OTHER = -1.0 - jnp.arange(12.0).reshape(4, 3)


_SCALE = jnp.asarray([1.0, 2.0, 3.0, 4.0])


def _branches(scale, traces=None):
    """Two branches over ``x`` that close over a float and an integer, counting their traces."""
    count = jnp.asarray(3)

    def true_fun(x):
        if traces is not None:
            traces["true"] += 1
        return jnp.sin(x) * scale + count, jnp.sum(x) * scale

    def false_fun(x):
        if traces is not None:
            traces["false"] += 1
        return jnp.cos(x) - scale, jnp.sum(x**2)

    return true_fun, false_fun


def _conditional(pred, x, scale):
    """``lane_cond`` of the two branches."""
    return lane_cond(pred, *_branches(scale), x)


def _per_lane(pred, x, scale):
    """The reference: both branches, then ``where`` on each lane."""
    taken, kept = (branch(x) for branch in _branches(scale))
    return jax.tree_util.tree_map(lambda t, f: jnp.where(pred, t, f), taken, kept)


# measured 8.9e-16 absolute and 1.8e-16 relative between lane_cond and the per-lane select (the
# conditional's boundary changes XLA's fusion), workstation 2026-11-06,
# pytest tests/contracts/test_lane_select.py
_ROUNDING = 1e-15


def _assert_same(got, want):
    """Every leaf of ``got`` equals ``want``'s to rounding."""
    for g, w in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)):
        np.testing.assert_allclose(g, w, rtol=_ROUNDING, atol=_ROUNDING)


def test_lane_cond_under_vmap_keeps_each_lane_branch_at_every_nesting():
    """Under vmap with lanes split between the branches, and with the predicate batched by
    the outer, the inner or both of two nested vmaps, each lane gets its own branch's result;
    a live lane of ``live_lane_cond`` does too."""
    outer = jnp.stack([_PRED, ~_PRED])
    xs, scales = jnp.stack([_TAKEN, _OTHER]), jnp.stack([_SCALE, _SCALE + 0.5])
    nestings = (
        (lambda f: jax.vmap(jax.vmap(f)), outer),
        (lambda f: jax.vmap(jax.vmap(f, (None, 0, 0))), outer[:, 0]),
        (lambda f: jax.vmap(jax.vmap(f), (None, 0, 0)), _PRED),
    )
    for nest, pred in nestings:
        _assert_same(
            jax.jit(nest(_conditional))(pred, xs, scales), nest(_per_lane)(pred, xs, scales)
        )
    live = jnp.asarray([True, True, False, True])

    def live_conditional(pred, alive, x, scale):
        return live_lane_cond(pred, *_branches(scale), x, live=alive)

    got = jax.vmap(live_conditional)(_PRED, live, _TAKEN, _SCALE)
    want = jax.vmap(_per_lane)(_PRED, _TAKEN, _SCALE)
    _assert_same(*(jax.tree_util.tree_map(lambda leaf: leaf[live], out) for out in (got, want)))


def test_lane_cond_derivatives_follow_each_lane_branch():
    """The forward and reverse derivatives of ``lane_cond``, unbatched, under vmap and through
    ``jax.checkpoint``, equal those of the per-lane select of both branches."""

    def loss(conditional):
        return lambda pred, x, scale: sum(jnp.sum(leaf) for leaf in conditional(pred, x, scale))

    def derivatives(conditional):
        def at(pred, x, scale):
            tangent = jax.jvp(lambda x, s: conditional(pred, x, s), (x, scale), (x + 1.0, 0.5))
            return tangent, jax.grad(jax.checkpoint(loss(conditional)), argnums=(1, 2))(
                pred, x, scale
            )

        return at

    for pred in (True, False):
        _assert_same(
            derivatives(_conditional)(pred, _TAKEN[0], 1.3),
            derivatives(_per_lane)(pred, _TAKEN[0], 1.3),
        )
    _assert_same(
        jax.vmap(derivatives(_conditional))(_PRED, _TAKEN, _SCALE),
        jax.vmap(derivatives(_per_lane))(_PRED, _TAKEN, _SCALE),
    )


def test_unbatched_lane_cond_is_one_conditional_with_each_branch_traced_once():
    """Unbatched, ``lane_cond`` compiles to one conditional; batched or not, each branch is
    traced once."""
    traces = {"true": 0, "false": 0}
    program = jax.jit(lambda p, x: lane_cond(p, *_branches(1.3, traces), x)).lower(True, _TAKEN[0])
    assert program.compile().as_text().count(" conditional(") == 1
    assert traces == {"true": 1, "false": 1}
    traces.update(true=0, false=0)
    jax.jit(jax.vmap(lambda p, x: lane_cond(p, *_branches(1.3, traces), x))).lower(_PRED, _TAKEN)
    assert traces == {"true": 1, "false": 1}
