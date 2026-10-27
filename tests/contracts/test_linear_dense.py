"""The linear-solver contract on its reference instance, the dense backend: every rule a
backend owes, proven once on a small nonsymmetric pattern with a duplicated entry (duplicates
add). The other backend files prove only what differs from this one."""

from __future__ import annotations

import copy
import pickle
from collections import Counter
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.linear import (
    Analysis,
    Dense,
    Factor,
    FactorContractError,
    SolveContext,
    SolveReport,
    SolveStatus,
    SparseTopology,
    analyze,
    empty_factor,
    factor,
    invalidate,
    refactor,
    solve,
    solve_checked,
)
from tests.support.programs import primitive_names

N = 8
DIRICHLET_ROWS = (1, 6)
# measured at most 2.1e-16 relative over the solves, gradients and Hessians below, CPU VM
# 2026-10-22,
# pytest tests/contracts/test_linear_dense.py
EXACT = {"rtol": 1e-12, "atol": 1e-13}


def _system(*, dirichlet=False, n=N, long_range=((0, 5), (2, 7), (6, 1), (4, 0)), seed=7):
    """``(topology, values)``: tridiagonal band, long-range couplings and a duplicate."""
    pairs = [(i, j) for i in range(n) for j in (i - 1, i, i + 1) if 0 <= j < n]
    pairs += [*long_range, (n // 2, n // 2)]
    rows, cols = (np.asarray(axis, np.int32) for axis in zip(*pairs))
    values = np.random.default_rng(seed).uniform(-1.0, 1.0, rows.size)
    values += np.where(rows == cols, 4.0, 0.0)
    bc = np.asarray(DIRICHLET_ROWS, np.int32) if dirichlet else None
    return SparseTopology(rows=rows, cols=cols, n_dofs=n, bc_indices=bc), jnp.asarray(values)


def _matrix(topology, values):
    """The operator a factor represents: duplicates add, Dirichlet rows are identity rows."""
    a = np.zeros((topology.n_dofs, topology.n_dofs))
    np.add.at(a, (topology.rows, topology.cols), np.asarray(values))
    if topology.bc_indices is not None:
        a[topology.bc_indices] = 0.0
        a[topology.bc_indices, topology.bc_indices] = 1.0
    return a


def _rhs(n=N, count=None):
    base = np.linspace(-1.0, 1.0, n) + 0.3 * np.cos(np.arange(n))
    if count is None:
        return jnp.asarray(base)
    return jnp.asarray(
        np.stack([(k + 1.0) * base + np.sin(k * np.arange(n)) for k in range(count)])
    )


def _reference(matrix, rhs):
    return np.linalg.solve(matrix, np.asarray(rhs).T).T


def _direction(values, seed=3):
    return values * jnp.asarray(np.random.default_rng(seed).uniform(-1.0, 1.0, values.shape))


def _lanes(values, count=4, seed=3):
    return jnp.stack([values + 0.05 * k * _direction(values, seed) for k in range(count)])


def _value_cotangent(topology, lam, x):
    """``-lam_i x_j`` on every pattern entry outside a Dirichlet row."""
    out = -lam[topology.rows] * x[topology.cols]
    if topology.bc_indices is None:
        return out
    return jnp.where(jnp.asarray(np.isin(topology.rows, topology.bc_indices)), 0.0, out)


def _counts(fn, *args):
    """``(factorizations, solves)`` bound in the program ``fn`` traces to."""
    names = Counter(primitive_names(fn, *args))
    return names["phased_factor"], names["phased_solve"]


@pytest.fixture(scope="module")
def system():
    topology, values = _system()
    return topology, values, analyze(topology, backend=Dense())


def test_analysis_is_memoised_by_content_and_refuses_undeclared_precisions(system):
    """An equal pattern built afresh gives an equal, hashable analysis without array leaves (a
    topology is content-hashed host data), each declared precision rung solves to its accuracy,
    and an undeclared one is refused."""
    topology, values, first = system
    second = analyze(_system()[0], backend=Dense())
    assert isinstance(first, Analysis) and second == first and hash(second) == hash(first)
    assert not any(isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves(first))
    assert (first.n, first.nnz, first.precision) == (N, topology.rows.size, "fp64")
    exact = np.linalg.solve(_matrix(topology, values), _rhs())
    # measured relative errors 4.5e-18 (fp64) and 3.7e-8 (fp32), CPU VM 2026-10-22
    for rung, bound in (("fp64", 1e-13), ("fp32", 1e-6)):
        f = factor(analyze(topology, backend=Dense(), precision=rung), values)
        error = np.max(np.abs(np.asarray(solve(f, _rhs())[0], np.float64) - exact))
        assert f.precision == rung and error / np.max(np.abs(exact)) <= bound
    with pytest.raises(ds.ConfigurationError, match="(?i)precision"):
        analyze(topology, backend=Dense(), precision="fp8")
    full = SparseTopology.full(3)
    listed = SparseTopology(
        rows=np.repeat(np.arange(3), 3), cols=np.tile(np.arange(3), 3), n_dofs=3
    )
    assert full == listed and hash(full) == hash(listed) and jax.tree_util.tree_leaves(full) == []


def test_factor_and_solve_match_numpy_in_both_orientations(system):
    """A factor crosses jit; its solves of one and many right-hand sides match NumPy for
    ``A`` and ``A^T``, and a factor of orientation "AT" represents the transpose."""
    topology, values, analysis = system
    a, b, block = _matrix(topology, values), _rhs(), _rhs(count=3)
    f = jax.jit(lambda v: factor(analysis, v))(values)
    f_t = factor(analysis, values, orientation="AT")
    assert f.orientation == "A" and f_t.orientation == "AT" and f_t.transposed
    run = jax.jit(solve, static_argnames="transpose")
    for held, transpose, matrix in (
        (f, False, a),
        (f, True, a.T),
        (f_t, False, a.T),
        (f_t, True, a),
    ):
        x, after_solve = run(held, b, transpose=transpose)
        np.testing.assert_allclose(x, _reference(matrix, b), **EXACT)
        np.testing.assert_array_equal(after_solve.generation, held.generation)
        np.testing.assert_allclose(
            run(after_solve, block, transpose=transpose)[0], _reference(matrix, block), **EXACT
        )
    b32 = b.astype(jnp.float32)
    x32 = run(f, b32, transpose=False)[0]
    assert x32.dtype == jnp.float64
    np.testing.assert_allclose(x32, _reference(a, np.asarray(b32, np.float64)), **EXACT)


def test_dirichlet_rows_are_identity_rows_whose_values_have_no_derivative():
    """Each Dirichlet row of the factored operator is an identity row in both orientations;
    the values stored in those rows change nothing and their derivative is zero."""
    topology, values = _system(dirichlet=True)
    analysis = analyze(topology, backend=Dense())
    in_bc = np.isin(topology.rows, DIRICHLET_ROWS)
    a, b = _matrix(topology, values), _rhs()
    x = solve(factor(analysis, values), b)[0]
    np.testing.assert_allclose(x, _reference(a, b), **EXACT)
    np.testing.assert_allclose(
        solve(factor(analysis, values), b, transpose=True)[0], _reference(a.T, b), **EXACT
    )
    rewritten = values + jnp.where(jnp.asarray(in_bc), 10.0, 0.0)
    np.testing.assert_allclose(solve(factor(analysis, rewritten), b)[0], x, **EXACT)
    gradient = np.asarray(
        jax.grad(lambda v: jnp.sum(solve(factor(analysis, v), b)[0] ** 2))(values)
    )
    np.testing.assert_array_equal(gradient[in_bc], 0.0)


def test_first_order_derivatives_match_the_closed_forms(system):
    """``dx = A^-1 (db - dA x)``; the cotangent of ``b`` is ``A^-T g`` and of the values
    ``-lam x^T`` on the pattern, so two entries at one position share their derivative."""
    topology, values, analysis = system
    a, b = _matrix(topology, values), _rhs()
    dv, db = _direction(values), jnp.cos(jnp.arange(N, dtype=jnp.float64))

    def solution(v, r):
        return solve(factor(analysis, v), r)[0]

    def loss(v, r):
        x = solution(v, r)
        return jnp.sum(jnp.sin(x)) + 0.5 * x @ x

    _, dx = jax.jit(lambda v, r: jax.jvp(solution, (v, r), (dv, db)))(values, b)
    x = np.linalg.solve(a, b)
    np.testing.assert_allclose(dx, np.linalg.solve(a, db - _matrix(topology, dv) @ x), **EXACT)
    gv, gb = jax.jit(jax.grad(loss, argnums=(0, 1)))(values, b)
    lam = np.linalg.solve(a.T, np.cos(x) + x)
    np.testing.assert_allclose(gb, lam, **EXACT)
    np.testing.assert_allclose(
        gv, _value_cotangent(topology, jnp.asarray(lam), jnp.asarray(x)), **EXACT
    )
    shared = np.flatnonzero((topology.rows == N // 2) & (topology.cols == N // 2))
    assert shared.size == 2 and gv[shared[0]] == gv[shared[1]]


def test_second_order_derivatives_compose(system):
    """The Hessian with respect to the right-hand side is ``A^-T diag(2x) A^-1``, and with
    respect to the values of a three-row system it is symmetric and matches differences."""
    topology, values, analysis = system
    a, b = _matrix(topology, values), _rhs()
    x, inverse = np.linalg.solve(a, b), np.linalg.inv(a)
    hessian_b = jax.jit(
        jax.hessian(lambda r: jnp.sum(solve(factor(analysis, values), r)[0] ** 3) / 3.0)
    )(b)
    np.testing.assert_allclose(hessian_b, inverse.T @ np.diag(2.0 * x) @ inverse, **EXACT)
    small, small_values = _system(n=3, long_range=((0, 2),))
    small_analysis = analyze(small, backend=Dense())

    def cubic(v):
        return jnp.sum(solve(factor(small_analysis, v), _rhs(3))[0] ** 3) / 3.0

    hessian_v = np.asarray(jax.jit(jax.hessian(cubic))(small_values))
    gradient, h = jax.jit(jax.grad(cubic)), 1e-6
    fd = np.stack(
        [
            (gradient(small_values + h * e) - gradient(small_values - h * e)) / (2 * h)
            for e in np.eye(small_values.size)
        ]
    )
    np.testing.assert_allclose(hessian_v, hessian_v.T, **EXACT)
    # measured 3.8e-10 relative, 3.3e-12 absolute against central differences, CPU VM 2026-10-22
    np.testing.assert_allclose(hessian_v, fd, rtol=1e-7, atol=1e-9)


def test_one_factorization_per_factor_and_orientation(system):
    """A program of several solves, a block solve, a tangent and a cotangent against one
    factor binds one factorization, under its gradient too; a second orientation one more; a
    vmap over right-hand sides of a held factor is one block solve and no factorization."""
    topology, values, analysis = system
    b = _rhs()

    def program(v, r):
        f = factor(analysis, v)
        x, f = solve(f, r)
        y, f = solve(f, x, transpose=True)
        block, f = solve(f, jnp.stack((r, x, y)))
        _, x_dot = jax.jvp(lambda rhs: solve(f, rhs)[0], (r,), (y,))
        return jnp.sum(x * y) + jnp.sum(block**2) + jnp.sum(x_dot)

    def both(v, r):
        f, f_t = factor(analysis, v), factor(analysis, v, orientation="AT")
        return jnp.sum(solve(f, r)[0] + solve(f_t, r, transpose=True)[0])

    assert _counts(program, values, b)[0] == 1
    assert _counts(jax.grad(program, argnums=(0, 1)), values, b)[0] == 1
    assert _counts(jax.grad(both, argnums=(0, 1)), values, b)[0] == 2
    a, f, batch = _matrix(topology, values), factor(analysis, values), _rhs(count=4)
    for transpose, matrix in ((False, a), (True, a.T)):
        mapped = jax.vmap(lambda r, t=transpose: solve(f, r, transpose=t)[0])
        assert _counts(mapped, batch) == (0, 1)
        np.testing.assert_allclose(jax.jit(mapped)(batch), _reference(matrix, batch), **EXACT)


def test_vmap_over_values_runs_as_one_batch_that_matches_every_lane(system):
    """Batched factors carry a lane axis on every leaf, a fused factor and solve binds one
    of each whatever the lane count, and lanes match NumPy and their serial gradients."""
    topology, values, analysis = system
    lanes, rhs_lanes, b = _lanes(values), _rhs(count=4), _rhs()
    factors = jax.jit(jax.vmap(lambda v: factor(analysis, v)))(lanes)
    assert all(leaf.shape[0] == 4 for leaf in jax.tree_util.tree_leaves(factors))
    fused = jax.vmap(lambda v, r: solve(factor(analysis, v), r)[0])
    assert _counts(fused, lanes, rhs_lanes) == (1, 1)
    out = jax.jit(fused)(lanes, rhs_lanes)

    def loss(v):
        return jnp.sum(jnp.sin(solve(factor(analysis, v), b)[0]))

    gradients = jax.jit(jax.vmap(jax.grad(loss)))(lanes)
    for k in range(4):
        np.testing.assert_allclose(
            out[k], _reference(_matrix(topology, lanes[k]), rhs_lanes[k]), **EXACT
        )
        np.testing.assert_allclose(gradients[k], jax.grad(loss)(lanes[k]), **EXACT)


def test_refactor_in_a_scan_keeps_the_factor_where_the_predicate_is_false(system):
    """A step whose predicate is false keeps the held factor, its values and generation; a
    true one factors the new values; the gradient reaches only values a factor was built from."""
    topology, values, analysis = system
    b, schedule = _rhs(), _lanes(values + 0.1 * _direction(values, seed=5))
    predicates = jnp.asarray([False, True, False, True])
    held = factor(analysis, values)

    def run(vs):
        def step(f, inputs):
            f = refactor(f, *inputs)
            x, f = solve(f, b)
            return f, (x, f.values, f.generation)

        return jax.lax.scan(step, held, (vs, predicates))

    final, (xs, carried, generations) = jax.jit(run)(schedule)
    np.testing.assert_array_equal(generations, [0, 1, 1, 2])
    for k, source in enumerate((values, schedule[1], schedule[1], schedule[3])):
        np.testing.assert_array_equal(carried[k], source)
        np.testing.assert_allclose(xs[k], _reference(_matrix(topology, source), b), **EXACT)
    through_loop = jax.jit(jax.grad(lambda vs: jnp.sum(run(vs)[1][0] ** 2)))(schedule)
    np.testing.assert_array_equal(through_loop[np.asarray([0, 2])], 0.0)


def test_checkpointed_factor_and_solve_give_the_same_gradient(system):
    """jax.checkpoint around the program, around each solve, or around a refactor keeps the
    derivative rules, and so does a user custom_vjp whose forward refactors, solves and
    invalidates and whose backward solves the transpose on the saved factor: every gradient
    equals that of the plain program."""
    topology, values, analysis = system
    b = _rhs()

    def loss(v, r):
        f = factor(analysis, v)
        x, f = solve(f, r)
        return jnp.sum(jnp.sin(x) * solve(f, x, transpose=True)[0])

    def per_solve(v, r):
        f = factor(analysis, v)
        x, f = jax.checkpoint(solve)(f, r)
        return jnp.sum(jnp.sin(x) * jax.checkpoint(partial(solve, transpose=True))(f, x)[0])

    def refactored(v, r):
        f = jax.checkpoint(lambda h, u: refactor(h, u, True))(factor(analysis, 1.1 * v), v)
        x, f = solve(f, r)
        return jnp.sum(jnp.sin(x) * solve(f, x, transpose=True)[0])

    held = factor(analysis, 1.1 * values)

    @jax.custom_vjp
    def wrapped(v, r):
        return solve(refactor(held, v, True), r)[0]

    def forward(v, r):
        x, f = solve(refactor(held, v, True), r)
        return x, (invalidate(f), x)

    def backward(saved, x_bar):
        f, x = saved
        lam = solve(f, x_bar, transpose=True)[0]
        return _value_cotangent(topology, lam, x), lam

    wrapped.defvjp(forward, backward)

    def custom(v, r):
        return jnp.sum(
            jnp.sin(wrapped(v, r)) * solve(factor(analysis, v), wrapped(v, r), transpose=True)[0]
        )

    expected = jax.grad(loss, argnums=(0, 1))(values, b)
    for variant in (jax.checkpoint(loss), per_solve, refactored, custom):
        for got, want in zip(jax.jit(jax.grad(variant, argnums=(0, 1)))(values, b), expected):
            np.testing.assert_allclose(got, want, **EXACT)


def test_a_checked_solve_reports_its_status_and_a_failed_factor_never_finite_output(system):
    """A zero row reports SINGULAR, a NaN reports NONFINITE, in both orientations under jit,
    with no finite entry for one or many right-hand sides; a regular matrix reports SUCCESS,
    an exact residual, zero iterations and one application, and an fp32 factor its larger
    residual."""
    topology, values, analysis = system
    singular = jnp.where(jnp.asarray(topology.rows == 5), 0.0, values)
    for broken, status in (
        (singular, SolveStatus.SINGULAR),
        (values.at[0].set(jnp.nan), SolveStatus.NONFINITE),
    ):
        f = jax.jit(lambda v: factor(analysis, v))(broken)
        for transpose in (False, True):
            x, _, report = jax.jit(partial(solve_checked, transpose=transpose))(f, _rhs())
            assert int(report.status) == status and not np.isfinite(np.asarray(x)).any()
            assert not np.isfinite(
                np.asarray(solve(f, _rhs(count=2), transpose=transpose)[0])
            ).any()
    for rhs in (_rhs(), _rhs(count=3)):
        _, _, report = jax.jit(solve_checked)(factor(analysis, values), rhs)
        assert isinstance(report, SolveReport) and int(report.status) == SolveStatus.SUCCESS
        # measured residual 5.6e-17, CPU VM 2026-10-22
        assert float(report.residual) < 1e-13
        assert (int(report.iterations), int(report.applications)) == (0, 1)
    coarse = factor(analyze(topology, backend=Dense(), precision="fp32"), values)
    # measured fp32 residual 6.0e-8, CPU VM 2026-10-22
    assert 1e-13 < float(solve_checked(coarse, _rhs())[2].residual) < 1e-5


def test_a_factor_is_a_static_pytree_that_outlives_its_context_and_refuses_malformed_input(system):
    """The leaves are the values, the handle, the generation and the int32 order word, the rest is
    hashable structure; factors survive pickle, deepcopy and their released context (which
    refuses new factors and pickling) and share one program across contexts; values that do
    not fit, an unknown orientation and an integer empty factor are refused."""
    _, values, analysis = system
    f = factor(analysis, values)
    expected = [f.values, *f.handle, f.generation, f.order]
    assert {id(leaf) for leaf in jax.tree_util.tree_leaves(f)} == {id(leaf) for leaf in expected}
    assert (f.generation.dtype, f.order.dtype, f.order.shape) == (jnp.int32, jnp.int32, ())
    structure = jax.tree_util.tree_structure(f)
    hash(structure)
    assert jax.tree_util.tree_structure(factor(analysis, 2.0 * values)) == structure
    assert jax.tree_util.tree_structure(factor(analysis, values, orientation="AT")) != structure
    assert jax.tree_util.tree_structure(jax.jit(lambda held: held)(f)) == structure
    for copied in (pickle.loads(pickle.dumps(f)), copy.deepcopy(f)):
        assert isinstance(copied, Factor) and jax.tree_util.tree_structure(copied) == structure
        np.testing.assert_array_equal(solve(copied, _rhs())[0], solve(f, _rhs())[0])
    assert pickle.loads(pickle.dumps(analysis)) == analysis == copy.deepcopy(analysis)
    with SolveContext() as ctx, pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(ctx)
    for call, match in (
        (lambda: factor(analysis, values[:-1]), None),
        (lambda: factor(analysis, values, orientation="A^H"), "orientation"),
        (lambda: refactor(f, values[:-1], jnp.asarray(True)), None),
        (lambda: empty_factor(analysis, dtype=jnp.int32), "dtype"),
    ):
        with pytest.raises(FactorContractError, match=match):
            call()
    traces = []
    run = jax.jit(lambda f, r: (traces.append(1), solve(f, r)[0])[1])
    with SolveContext() as ctx:
        f = factor(analysis, values, ctx=ctx)
        x = run(f, _rhs())
        assert ctx.resident_bytes() == 0
    np.testing.assert_array_equal(run(f, _rhs()), x)
    with pytest.raises(FactorContractError, match="released"):
        factor(analysis, values, ctx=ctx)
    ctx.release()
    with SolveContext() as other:
        np.testing.assert_array_equal(run(factor(analysis, values, ctx=other), _rhs()), x)
    assert len(traces) == 1


def test_invalidation_keeps_its_place_and_an_empty_factor_holds_nothing_until_filled(system):
    """An invalidation takes the factorization's order word and its result feeds the next solve,
    the program carries no effect and stays exact, and a refactor, solve and invalidation
    compiled ahead of time run on every call with the values in force; an empty factor has a
    built factor's structure at generation -1, its solves report EMPTY with no finite entry,
    a false refactor keeps it empty and a true one fills it."""
    topology, values, analysis = system

    def program(v, r):
        return solve(invalidate(factor(analysis, v)), r)[0]

    jaxpr = jax.make_jaxpr(program)(values, _rhs())
    assert not jaxpr.effects
    made, mark, solved = (
        next(e for e in jaxpr.jaxpr.eqns if e.primitive.name == n)
        for n in ("phased_factor", "phased_invalidate", "phased_solve")
    )
    assert mark.invars[-1] is made.outvars[-1] and mark.outvars[0] in solved.invars
    np.testing.assert_allclose(
        jax.jit(program)(values, _rhs()), _reference(_matrix(topology, values), _rhs()), **EXACT
    )
    held = factor(analysis, values)

    def gated(f, v, predicate):
        x, g = solve(refactor(f, v, predicate), _rhs())
        return x, invalidate(g)

    compiled = jax.jit(gated).lower(held, values, jnp.asarray(True)).compile()
    for scale, predicate in ((1.2, True), (0.9, False), (0.8, True)):
        x, invalidated = compiled(held, scale * values, jnp.asarray(predicate))
        source = scale * values if predicate else values
        assert int(invalidated.generation) == int(predicate)
        np.testing.assert_allclose(x, _reference(_matrix(topology, source), _rhs()), **EXACT)
    empty = invalidate(empty_factor(analysis))
    assert jax.tree_util.tree_structure(empty) == jax.tree_util.tree_structure(
        factor(analysis, values)
    )
    assert int(empty.generation) == -1
    for transpose in (False, True):
        x, _, report = jax.jit(partial(solve_checked, transpose=transpose))(empty, _rhs())
        assert int(report.status) == SolveStatus.EMPTY and not np.isfinite(np.asarray(x)).any()
    assert int(jax.jit(refactor)(empty, values, jnp.asarray(False)).generation) == -1
    filled = jax.jit(refactor)(empty, values, jnp.asarray(True))
    assert int(filled.generation) == 0
    np.testing.assert_allclose(
        solve(filled, _rhs())[0], _reference(_matrix(topology, values), _rhs()), **EXACT
    )
