"""Sparse-matrix colorings: each kind and order decompresses its exact compressed products into
every entry under ``jit``, every coloring is structurally valid, a pattern a kind cannot color
is refused, and the tangents and pullbacks a coloring seeds recover the Jacobian."""

from __future__ import annotations

import itertools
import math

import jax
import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from diff_soup.problem._ad.coloring import (
    KINDS,
    MAX_PAIRS,
    ORDERS,
    color,
    colored_values,
    decompress,
)

_SYMMETRIC = ("star", "acyclic")
_CASES = list(itertools.product(KINDS, ORDERS))


def _patterns(kind):
    """Three random patterns, entries shuffled, some lines empty; symmetric with part of the
    diagonal for the symmetric kinds."""
    rng = np.random.default_rng(KINDS.index(kind))
    for m, n, density in ((17, 23, 0.15), (30, 30, 0.08), (9, 40, 0.3)):
        pattern = rng.random((n, n) if kind in _SYMMETRIC else (m, n)) < density
        if kind in _SYMMETRIC:
            pattern |= pattern.T
            pattern[np.diag_indices(n)] = rng.random(n) < 0.5
        rows, cols = np.nonzero(pattern)
        shuffle = rng.permutation(rows.size)
        yield rows[shuffle], cols[shuffle], pattern.shape


def _counts(pattern, colors, n_colors):
    """Per row of ``pattern``, its entries in the columns of each color; column 0 is no color."""
    return pattern.astype(int) @ (colors[:, None] == np.arange(-1, n_colors))


def _valid(rows, cols, shape, coloring):
    """SMC ``check.jl``: every entry is alone in its row among its column's color or in its
    column among its row's color (the row's own column color for a symmetric kind), and
    colors run ``0, 1, ...``; an acyclic coloring instead two-colors only forests, with its
    diagonal colored and an entry by a vertex of no color alone as above."""
    pattern = np.zeros(shape, dtype=bool)
    pattern[rows, cols] = True
    cc = coloring.column_colors
    rc = cc if coloring.kind in _SYMMETRIC else coloring.row_colors
    n_rc = coloring.n_forward if coloring.kind in _SYMMETRIC else coloring.n_reverse
    alone = (
        (cc[cols] >= 0) & (_counts(pattern, cc, coloring.n_forward)[rows, cc[cols] + 1] == 1)
    ) | ((rc[rows] >= 0) & (_counts(pattern.T, rc, n_rc)[cols, rc[rows] + 1] == 1))
    contiguous = set(cc[cc >= 0]) == set(range(coloring.n_forward)) and set(
        coloring.row_colors[coloring.row_colors >= 0]
    ) == set(range(coloring.n_reverse))
    if coloring.kind != "acyclic":
        return contiguous and bool(np.all(alone))
    off, colored = rows != cols, (cc[rows] >= 0) & (cc[cols] >= 0)
    forests = True
    for a, b in itertools.combinations(range(coloring.n_forward), 2):
        keep = np.isin(cc, (a, b))
        sub = pattern[np.ix_(keep, keep)] & ~np.eye(int(keep.sum()), dtype=bool)
        edges = int(sub.sum()) // 2
        forests &= edges == keep.sum() - connected_components(sub, directed=False)[0]
    return (
        contiguous
        and forests
        and not np.any(off & colored & (cc[rows] == cc[cols]))
        and bool(np.all(cc[rows[~off]] >= 0))
        and bool(np.all(alone[off & ~colored]))
    )


@pytest.mark.parametrize("kind, order", _CASES)
def test_decompressing_exact_products_recovers_every_entry(kind, order):
    """``decompress`` under ``jit`` reads every entry back from the products of a random matrix
    (a symmetric one for the symmetric kinds) with the seeds the coloring gives."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for rows, cols, shape in _patterns(kind):
        matrix = np.zeros(shape)
        matrix[rows, cols] = rng.normal(size=rows.size)
        if kind in _SYMMETRIC:
            matrix = matrix + matrix.T
        coloring = color(rows, cols, shape, kind, order)
        forward = (coloring.column_colors == np.arange(coloring.n_forward)[:, None]) @ matrix.T
        reverse = (coloring.row_colors == np.arange(coloring.n_reverse)[:, None]) @ matrix
        values = jax.jit(lambda f, r, c=coloring: decompress(c, f, r))(forward, reverse)
        worst = max(worst, float(np.max(np.abs(values - matrix[rows, cols]), initial=0.0)))
    # measured 7.8e-16, CPU 2026-11-06, pytest tests/contracts/test_coloring.py
    assert worst <= 1e-12


@pytest.mark.parametrize("kind, order", _CASES)
def test_every_coloring_is_structurally_valid(kind, order):
    """Each coloring of each random pattern passes its kind's structural check."""
    assert all(_valid(r, c, s, color(r, c, s, kind, order)) for r, c, s in _patterns(kind))


_DENSE = math.isqrt(MAX_PAIRS) + 1
_REFUSED = {
    "star, not symmetric": ([0, 1], [1, 1], (2, 2), "star", "structurally symmetric"),
    "acyclic, not symmetric": ([0, 1], [1, 1], (2, 2), "acyclic", "structurally symmetric"),
    "column, a row past MAX_PAIRS": (
        [0] * _DENSE,
        range(_DENSE),
        (1, _DENSE),
        "column",
        "MAX_PAIRS",
    ),
}


@pytest.mark.parametrize("name", sorted(_REFUSED))
def test_a_pattern_the_kind_cannot_color_is_refused(name):
    """``star`` and ``acyclic`` refuse a pattern that is not structurally symmetric, and a column
    coloring refuses a dense row whose pairs pass ``MAX_PAIRS``."""
    rows, cols, shape, kind, match = _REFUSED[name]
    with pytest.raises(ValueError, match=match):
        color(np.asarray(rows), np.asarray(list(cols)), shape, kind)


@pytest.mark.parametrize("kind", ("column", "row", "bidirectional"))
def test_the_tangents_and_pullbacks_a_coloring_seeds_recover_the_jacobian(kind):
    """The linearization plan's whole-pattern route: the colored tangents and pullbacks of a
    nonlinear map with a random pattern give every entry ``jax.jacfwd`` gives, to 1e-12."""
    rng = np.random.default_rng(7)
    mask = rng.random((24, 24)) < 0.12
    mask[np.diag_indices(24)] = True
    mask[3] = True
    rows, cols = np.nonzero(mask)
    weights = jax.numpy.asarray(np.where(mask, rng.uniform(0.5, 1.5, mask.shape), 0.0))

    def fn(x):
        return weights @ jax.numpy.sin(x) + x**2

    x = jax.numpy.asarray(rng.uniform(-1.0, 1.0, 24))
    coloring = color(rows, cols, mask.shape, kind=kind)
    values = jax.jit(lambda x: colored_values(coloring, fn, x))(x)
    expected = jax.jacfwd(fn)(x)[rows, cols]
    # measured 2.2e-16, CPU 2026-11-06, pytest tests/contracts/test_coloring.py
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-12)
