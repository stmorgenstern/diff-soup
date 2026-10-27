"""Linear contract, structured backends: block tridiagonal chains, Woodbury low-rank terms and
member blocks; their layouts, refusals and statuses, and solves equal to the dense ones."""

from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.linear import (
    KLU,
    BlockTridiagonal,
    Dense,
    LowRankTerm,
    MemberBlock,
    MemberBlockLayout,
    SolveContext,
    SolveStatus,
    SparseTopology,
    Woodbury,
    analyze,
    factor,
    refactor,
    solve,
    solve_checked,
)
from diff_soup.problem._ir._low_rank import (
    MemberMeanCoupling,
    member_block_layout,
    member_mean_term,
    member_mean_values,
)

# measured at most 2.1e-13 relative over the band, Woodbury (solves and the U gradient) and
# member-block solves, CPU VM 2026-10-22, pytest tests/contracts/test_linear_structured.py
EXACT = {"rtol": 1e-10, "atol": 1e-12}


#: Compiled once per analysis and shape: eager structured solves would compile per operation.
_factor = jax.jit(factor, static_argnums=0, static_argnames=("orientation",))
_checked = jax.jit(solve_checked, static_argnames=("transpose",))


def _solve(f, b, transpose=False):
    return _checked(f, b, transpose=transpose)[0]


def _banded(n, b, *, bc=None, seed=3, fill=0.4):
    """``(topology, values, dense)`` of a pattern inside the block band of size ``b``."""
    rng = np.random.default_rng(seed)
    entries = {
        (i, j): rng.uniform(-1, 1) + (2.0 * b + 4.0 if i == j else 0.0)
        for i in range(n)
        for j in range(n)
        if abs(i // b - j // b) <= 1 and (i // b == j // b or rng.uniform() < fill)
    }
    rows, cols = (np.asarray(axis) for axis in zip(*sorted(entries)))
    values = np.asarray([entries[key] for key in sorted(entries)])
    a = np.zeros((n, n))
    a[rows, cols] = values
    if bc is not None:
        a[bc] = 0.0
        a[bc, bc] = 1.0
    return SparseTopology(rows=rows, cols=cols, n_dofs=n, bc_indices=bc), jnp.asarray(values), a


def test_a_block_tridiagonal_backend_solves_padded_dirichlet_bands_and_refuses_the_rest():
    """A band pattern of 17 unknowns in blocks of 3 is padded, keeps its Dirichlet rows,
    solves both directions from one factor, batches over values with a gated refactor;
    a zero row and a NaN report SINGULAR and NONFINITE; an entry outside the band and an
    invalid block size are refused."""
    topology, values, a = _banded(17, 3, bc=np.asarray([0, 9]))
    analysis = analyze(topology, backend=BlockTridiagonal(3))
    rhs = jnp.asarray(np.random.default_rng(5).uniform(-1, 1, (2, 17)))
    f = _factor(analysis, values)
    for transpose, matrix in ((False, a), (True, a.T)):
        x, _, report = _checked(f, rhs, transpose=transpose)
        assert int(report.status) == SolveStatus.SUCCESS
        np.testing.assert_allclose(x, np.linalg.solve(matrix, rhs.T).T, **EXACT)
    kept = jax.jit(
        lambda vs, ws, p: jax.vmap(refactor)(jax.vmap(lambda v: factor(analysis, v))(vs), ws, p)
    )(jnp.stack([values, 1.3 * values]), jnp.stack([2.0 * values] * 2), jnp.asarray([False, True]))
    np.testing.assert_array_equal(kept.values[0], values)
    for broken, status in (
        (jnp.where(jnp.asarray(topology.rows == 4), 0.0, values), SolveStatus.SINGULAR),
        (values.at[3].set(jnp.inf), SolveStatus.NONFINITE),
    ):
        x, _, report = _checked(_factor(analysis, broken), jnp.ones(17))
        assert int(report.status) == status and not np.isfinite(np.asarray(x)).any()
    wide = SparseTopology(rows=[0, 1, 2, 0], cols=[0, 1, 2, 2], n_dofs=3)
    with pytest.raises(
        ds.ConfigurationError, match=r"entry \(0, 2\) joins block row 0 to block column 2"
    ):
        analyze(wide, backend=BlockTridiagonal(1))
    for bad in (0, -2, 2.5, True):
        with pytest.raises((TypeError, ValueError, ds.ConfigurationError), match="block_size"):
            BlockTridiagonal(bad)


N = 18


def _term(n=N):
    """A rank-3 term: two weighted means and one single-row coupling."""
    return LowRankTerm(
        n_dofs=n,
        rank=3,
        u_rows=[0, 1, 2, 3, 10, 11, 12, 5],
        u_cols=[0, 0, 0, 0, 1, 1, 1, 2],
        w_rows=[6, 7, 8, 9, 0, 1, 17, 14],
        w_cols=[0, 0, 0, 0, 1, 1, 1, 2],
        w_values=[0.25, 0.25, 0.25, 0.25, -0.5, 0.3, 0.2, 1.0],
    )


def _low_rank(bc=None, seed=2):
    """``(term, topology, values, dense)`` of a banded base plus ``U W^T``."""
    base, base_values, a = _banded(N, 3, bc=None, seed=seed, fill=1.0)
    term = _term()
    u = np.random.default_rng(seed + 5).uniform(-0.8, 0.8, term.n_u)
    a = a + np.asarray(term.dense(jnp.asarray(u)))
    if bc is not None:
        a[bc] = 0.0
        a[bc, bc] = 1.0
        base = SparseTopology(rows=base.rows, cols=base.cols, n_dofs=N, bc_indices=bc)
    return term, term.topology(base), jnp.concatenate([base_values, jnp.asarray(u)]), a


def test_a_low_rank_term_and_woodbury_validate_their_declarations_and_report_singular_bases():
    """A term hashes by content, refuses malformed supports and applies ``U W^T``; Woodbury
    refuses an inexact or contract-less inner backend and a topology not built from its term,
    and a singular capacitance or base reports SINGULAR with no finite entry."""
    term = _term()
    assert term == _term() and hash(term) == hash(_term()) and not jax.tree_util.tree_leaves(term)
    good = dict(n_dofs=4, rank=1, u_rows=[0], u_cols=[0], w_rows=[1], w_cols=[0], w_values=[1.0])
    for change, match in (
        (dict(u_rows=[0, 0], u_cols=[0, 0]), "twice"),
        (dict(rank=2), "coupling 1 has no entry"),
        (dict(w_rows=[4]), "outside"),
        (dict(w_values=[np.nan]), "finite"),
    ):
        with pytest.raises(ds.ConfigurationError, match=match):
            LowRankTerm(**{**good, **change})
    _, topology, values, _ = _low_rank()
    u = term.split(values)[1]
    x = jnp.asarray(np.random.default_rng(0).uniform(-1, 1, (2, N)))
    np.testing.assert_allclose(term.apply(u, x), x @ np.asarray(term.dense(u)).T, **EXACT)

    class Inexact(Dense):
        is_exact = False

    with pytest.raises(ds.ConfigurationError, match="exact inner backend"):
        Woodbury(Inexact(), term)
    with pytest.raises(ds.ConfigurationError, match="LinearSolver"):
        Woodbury(object(), term)
    with pytest.raises(ds.ConfigurationError, match="analyze LowRankTerm.topology"):
        analyze(term.base_topology(topology), backend=Woodbury(Dense(), term))
    # Row and column 4 hold only the diagonal 2 and the term adds u at (4, 4): u = -2 makes the
    # capacitance 1 + u / 2 zero; a zero base diagonal with u = 2 leaves the base singular.
    base = SparseTopology(
        rows=[0, 1, 2, 3, 4, 5, 0, 1, 2, 5], cols=[0, 1, 2, 3, 4, 5, 1, 2, 3, 0], n_dofs=6
    )
    single = LowRankTerm(
        n_dofs=6, rank=1, u_rows=[4], u_cols=[0], w_rows=[4], w_cols=[0], w_values=[1.0]
    )
    analysis = analyze(single.topology(base), backend=Woodbury(Dense(), single))
    capacitance = jnp.asarray([2.0] * 6 + [0.5, -0.3, 0.2, 0.1, -2.0])
    for broken in (capacitance, capacitance.at[4].set(0.0).at[-1].set(2.0)):
        x, _, report = _checked(_factor(analysis, broken), jnp.ones(6))
        assert int(report.status) == SolveStatus.SINGULAR and not np.isfinite(np.asarray(x)).any()


@pytest.mark.parametrize("inner", ["dense", pytest.param("klu", marks=pytest.mark.requires_klu)])
def test_woodbury_solves_both_directions_from_one_factor_and_differentiates_the_term(inner):
    """Woodbury over an exact inner backend takes the inner backend's residency, pickles by
    content, solves the whole operator (Dirichlet rows applied) in both directions from one
    factor, and its gradient of an entry of U is ``-lam_i (W^T x)_k``."""
    inner_backend = {"dense": Dense, "klu": KLU}[inner]()
    backend = Woodbury(inner_backend, _term())
    assert type(backend).factor_residency == type(inner_backend).factor_residency
    assert pickle.loads(pickle.dumps(backend)) == backend
    term, topology, values, a = _low_rank(bc=np.asarray([2, 11]))
    analysis = analyze(topology, backend=backend)
    rhs = jnp.asarray(np.random.default_rng(7).uniform(-1, 1, (3, N)))
    with SolveContext() as ctx:
        for orientation, matrix in (("A", a), ("AT", a.T)):
            f = factor(analysis, values, orientation=orientation, ctx=ctx)
            for transpose, oriented in ((False, matrix), (True, matrix.T)):
                np.testing.assert_allclose(
                    solve(f, rhs, transpose=transpose)[0],
                    np.linalg.solve(oriented, rhs.T).T,
                    **EXACT,
                )
        term, topology, values, a = _low_rank()
        analysis = analyze(topology, backend=Woodbury(inner_backend, term))
        b = jnp.linspace(-1.0, 1.0, N)

        def loss(v):
            y = solve(factor(analysis, v, ctx=ctx), b)[0]
            return jnp.sum(jnp.sin(y)) + 0.5 * y @ y

        gradient = jax.grad(loss)(values)
    x = np.linalg.solve(a, b)
    lam = np.linalg.solve(a.T, np.cos(x) + x)
    w_t_x = np.asarray(term.w_transpose_apply(jnp.asarray(x)))
    np.testing.assert_allclose(
        gradient[-term.n_u :], -lam[term.u_rows] * w_t_x[term.u_cols], **EXACT
    )


def _members():
    """Three field-major members (2, 3, 4 sites, 2 fields), one mean each, 2 retained."""
    rng = np.random.default_rng(11)
    sizes = (2, 3, 4)
    starts = 2 + np.concatenate([[0], np.cumsum([2 * n for n in sizes])[:-1]])
    n = 2 + 2 * sum(sizes)
    entries = {(0, 0): 4.0, (1, 1): 5.0, (0, 1): 0.5, (1, 0): -0.5}
    readers, groups = [], []
    for k, (start, size) in enumerate(zip(starts, sizes)):
        c, s = start + np.arange(size), start + size + np.arange(size)
        for i in range(size):
            entries[(c[i], c[i])] = rng.uniform(2.0, 3.0) * (-1.0 if k == 1 else 1.0)
            entries[(s[i], s[i])] = 1.0
            entries[(s[i], c[i])] = rng.uniform(-1.0, 1.0)
            entries[(c[i], k % 2)] = rng.uniform(-0.5, 0.5)
            if i:
                entries[(c[i], c[i - 1])] = rng.uniform(-1.0, 1.0)
                entries[(c[i - 1], c[i])] = rng.uniform(-1.0, 1.0)
                entries[(s[i], s[i - 1])] = -1.0
        entries[(k % 2, s[-1])] = rng.uniform(-1.0, -0.5)
        readers.append(np.concatenate([c, s]))
        groups.append(c)
    term = LowRankTerm.group_means(n, readers=readers, groups=groups)
    keys = sorted(entries)
    base = SparseTopology(rows=[r for r, _ in keys], cols=[c for _, c in keys], n_dofs=n)
    u = rng.uniform(-1.0, 1.0, term.n_u)
    dense = np.zeros((n, n))
    np.add.at(dense, (base.rows, base.cols), [entries[key] for key in keys])
    dense = dense + np.asarray(term.dense(jnp.asarray(u)))
    layout = MemberBlockLayout.contiguous(starts, sizes, 2, n_dofs=n, term=term)
    values = jnp.asarray(np.concatenate([[entries[key] for key in keys], u]))
    return layout, term.topology(base), values, dense


def test_a_member_block_layout_validates_and_its_backend_solves_or_fails_loudly():
    """A layout refuses overlapping members and an incomplete retained set; the backend refuses
    a topology of another size; it solves both directions exactly; a
    backward-error check no solve meets returns NaN with SINGULAR."""
    layout, topology, values, dense = _members()
    assert (layout.n_members, layout.n_chain, layout.block_size) == (3, 9, 2)
    for members, kwargs, match in (
        ([[[0, 1]], [[1, 2]]], {}, "disjoint"),
        ([[[0, 1]]], {"retained": [2]}, "unknown 3 is in neither"),
    ):
        with pytest.raises(ds.ConfigurationError, match=match):
            MemberBlockLayout(members, n_dofs=4, **kwargs)
    with pytest.raises(ds.ConfigurationError, match="size 21"):
        analyze(
            SparseTopology(rows=[0], cols=[0], n_dofs=topology.n_dofs + 1),
            backend=MemberBlock(layout),
        )
    analysis = analyze(topology, backend=MemberBlock(layout))
    rhs = jnp.linspace(-1.0, 1.0, topology.n_dofs)
    for transpose, matrix in ((False, dense), (True, dense.T)):
        np.testing.assert_allclose(
            _solve(_factor(analysis, values), rhs, transpose=transpose),
            np.linalg.solve(matrix, rhs),
            **EXACT,
        )
    strict = analyze(
        topology, backend=MemberBlock(layout, residual_tolerance=1e-300, refinement_sweeps=2)
    )
    x, _, report = _checked(_factor(strict, values), rhs)
    assert int(report.status) == SolveStatus.SINGULAR and not np.isfinite(np.asarray(x)).any()


def test_a_family_member_mean_coupling_gives_the_member_term_and_layout():
    """For a recipe family whose members read the mean of their field ``c``, the term and the
    layout built from the family's view, with U from one forward derivative along per-member
    offsets, reproduce the Jacobian, and MemberBlock solves it exactly."""
    spec = ds.problem("member_means").domain(
        "cell", ds.cartesian_grid_1d(2, 0.0, 1.0, method="fvm")
    )
    spec.field("m", mass="differential", scale=1.0)
    spec.equation(writes="m", reads=("m",))(lambda q: -q.m)
    members = spec.family(
        "p",
        count=3,
        domain=ds.cartesian_grid_1d(4, 0.0, 1.0, method="fvm"),
        scale={"c": 1.0, "s": 1.0},
        residual_scale={"c": 1.0, "s": 1.0},
    )
    members.field_decl("c", mass="differential")
    members.field_decl("s", mass="differential")
    members.equation(writes="c", reads=("c",))(lambda q, member: -q.c)
    members.equation(writes="s", reads=("s",))(lambda q, member: -q.s)
    view, n = spec.build().family_view("p"), 26
    assert (view.base, view.count) == (2, 3)

    def residual(y, offsets):
        members = y[2:].reshape(3, 2, 4)
        c, s = members[:, 0], members[:, 1]
        mean = jnp.mean(c, axis=1) + offsets
        left = jnp.pad(c, ((0, 0), (1, 0)))[:, :-1]
        r_c = 3.0 * c - 0.5 * left + 0.3 * mean[:, None] ** 2 + 0.2 * y[0]
        r_s = s - c + 0.1 * jnp.pad(s, ((0, 0), (1, 0)))[:, :-1] + 0.25 * mean[:, None]
        heads = jnp.stack([4.0 * y[0] + 0.1 * jnp.sum(c[:, -1]), 5.0 * y[1] + 0.2 * y[0]])
        return jnp.concatenate([heads, jnp.stack([r_c, r_s], 1).reshape(-1)])

    y, zero = jnp.asarray(np.random.default_rng(4).uniform(0.5, 1.5, n)), jnp.zeros(3)
    jacobian = np.asarray(jax.jacfwd(residual)(y, zero))
    term = member_mean_term(view, MemberMeanCoupling("p", "c"), n_dofs=n)
    u = member_mean_values(
        view,
        MemberMeanCoupling("p", "c"),
        jax.jvp(lambda o: residual(y, o), (zero,), (jnp.ones(3),))[1][None, :],
    )
    stencil = jacobian - np.asarray(term.dense(u))
    rows, cols = np.nonzero(stencil)
    layout = member_block_layout(view, ("c", "s"), n_dofs=n, term=term)
    np.testing.assert_array_equal(layout.retained, [0, 1])
    analysis = analyze(
        term.topology(SparseTopology(rows=rows, cols=cols, n_dofs=n)), backend=MemberBlock(layout)
    )
    b = jnp.linspace(-1.0, 1.0, n)
    x = _solve(_factor(analysis, term.join(jnp.asarray(stencil[rows, cols]), u)), b)
    np.testing.assert_allclose(x, np.linalg.solve(jacobian, b), **EXACT)
