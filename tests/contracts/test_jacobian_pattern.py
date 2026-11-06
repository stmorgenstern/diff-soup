"""Jacobian patterns read from a tangent map: each rule holds exactly the entries ``jax.jacfwd``
finds nonzero at random states, a branch or loop of unknown outcome holds every one of them,
an operation no rule covers falls back to every entry with a warning, and ``within`` bounds
the rows a fallback reaches."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.utils.lanes import lane_where

_N = 12
_KNOWN = jnp.eye(_N) * 2.0 + jnp.diag(jnp.ones(_N - 1), 1)
_HALF = jnp.asarray([0.0, 1.0] * (_N // 2))


def _reached(fn):
    """Entries ``jax.jacfwd`` finds nonzero at three random states."""
    rng = np.random.default_rng(1)
    seen = None
    for _ in range(3):
        jacobian = np.asarray(jax.jacfwd(fn)(jnp.asarray(rng.uniform(0.5, 1.5, _N)))) != 0
        seen = jacobian if seen is None else seen | jacobian
    return seen.reshape(-1, _N)


def _pattern(fn, shape):
    """The walk's pattern as a boolean matrix."""
    rows, cols = ds.jacobian_pattern(fn, jnp.zeros(_N))
    pattern = np.zeros(shape, dtype=bool)
    pattern[rows, cols] = True
    return pattern


def _scan(y):
    """A two-step loop over a static trip count, which JAX writes as a scan."""
    return jax.lax.fori_loop(0, 2, lambda i, v: v * v + 0.1 * jnp.roll(v, 1), y)


_EXACT = {
    "elementwise": lambda y: jnp.sin(y) / (1.0 + y**2) - y,
    "mul by a known zero": lambda y: y * _HALF + jnp.roll(y, 1) ** 2,
    "select_n on a known condition": lambda y: jnp.where(
        jnp.arange(_N) % 3 == 0, y**2, jnp.roll(y, 1)
    ),
    "moves": lambda y: jnp.concatenate([y.reshape(3, 4).T.ravel()[::-1][:6], jnp.tile(y[3:5], 2)]),
    "unstack": lambda y: jnp.stack(jax.lax.unstack(y.reshape(3, 4) ** 2, axis=0)[::-1]).ravel(),
    "split": lambda y: jnp.concatenate(jnp.split(y**2, [5])[::-1]),
    "gather at known indices": lambda y: y[jnp.asarray([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8])] ** 2,
    "dynamic_slice": lambda y: jax.lax.dynamic_slice(y**2, (jnp.int32(4),), (6,)),
    "dynamic_update_slice": lambda y: jax.lax.dynamic_update_slice(y, y[:4] ** 2, (jnp.int32(5),)),
    "pad": lambda y: jax.lax.pad(y**2, 0.0, ((2, -3, 1),)),
    "scatter-add": lambda y: y.at[jnp.asarray([0, 2, 2])].add(y[3:6] ** 2),
    "scatter": lambda y: y.at[jnp.asarray([0, 2, 7])].set(y[3:6] ** 2),
    "reduce_sum": lambda y: jnp.sum(y.reshape(3, 4) ** 2, axis=1),
    "cumsum": lambda y: jnp.cumsum(y**2),
    "dot_general with a known matrix": lambda y: _KNOWN @ y**2,
    "cond on a known index": lambda y: jax.lax.switch(
        1, (jnp.sin, lambda v: jnp.roll(v, 1) ** 2), y
    ),
    "scan": _scan,
    "custom_linear_solve": lambda y: jnp.linalg.solve(_KNOWN, y**2),
    "diffsoup_lane_where": lambda y: lane_where(y[0] > 1.0, y**2, jnp.roll(y, 1)),
    "conv_general_dilated": lambda y: jnp.convolve(
        y**2, jnp.asarray([1.0, -2.0, 1.0]), mode="same"
    ),
}
_UNKNOWN_OUTCOME = {
    "cond on the state": lambda y: jax.lax.cond(y[0] > 1.0, jnp.sin, lambda v: jnp.roll(v, 1), y),
    "while": lambda y: jax.lax.while_loop(
        lambda c: c[0] < 2, lambda c: (c[0] + 1, c[1] * c[1] + 0.1 * jnp.roll(c[1], 1)), (0, y)
    )[1],
}
_FALLBACK = {
    "gather at an index of the state": lambda y: jnp.sort(y) + y,
    "a primitive with no rule": lambda y: jnp.fft.fft(y).real,
}


@pytest.mark.parametrize("name", sorted(_EXACT))
def test_each_rule_holds_exactly_the_entries_jacfwd_finds(name):
    """The pattern of each rule's toy equals the entries ``jax.jacfwd`` finds nonzero."""
    reached = _reached(_EXACT[name])
    np.testing.assert_array_equal(_pattern(_EXACT[name], reached.shape), reached)


@pytest.mark.parametrize("name", sorted(_UNKNOWN_OUTCOME))
def test_a_branch_or_loop_of_unknown_outcome_holds_every_entry_jacfwd_finds(name):
    """A branch on the state takes both sides, and a while loop's carry its fixed point: the
    pattern holds every entry ``jax.jacfwd`` finds nonzero."""
    reached = _reached(_UNKNOWN_OUTCOME[name])
    assert not np.any(reached & ~_pattern(_UNKNOWN_OUTCOME[name], reached.shape))


@pytest.mark.parametrize("name", sorted(_FALLBACK))
def test_a_fallback_warns_and_holds_every_entry_jacfwd_finds(name):
    """An index the state sets and a primitive with no rule warn, and the pattern holds every
    entry ``jax.jacfwd`` finds nonzero."""
    reached = _reached(_FALLBACK[name])
    with pytest.warns(UserWarning, match="jacobian_pattern"):
        pattern = _pattern(_FALLBACK[name], reached.shape)
    assert not np.any(reached & ~pattern)


def test_within_bounds_only_the_rows_a_fallback_reaches():
    """Sorting the state makes its six rows read all twelve columns; with ``within`` the
    six keep the first six columns, and the six squared rows keep their diagonal."""
    within = (np.repeat(np.arange(6), 6), np.tile(np.arange(6), 6))
    with pytest.warns(UserWarning, match="jacobian_pattern"):
        rows, cols = ds.jacobian_pattern(
            lambda y: jnp.concatenate([jnp.sort(y)[:6], y[6:] ** 2]), jnp.zeros(_N), within=within
        )
    expected = sorted([*zip(*within), *((row, row) for row in range(6, _N))])
    assert list(zip(rows.tolist(), cols.tolist())) == expected
