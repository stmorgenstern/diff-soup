"""Linear contract, Krylov: restarted GMRES and its statuses, error weights and initial guesses,
matrix-free operators, and the Krylov backend's inexact factors on lagged preconditioners."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.linear import (
    Dense,
    FactorContractError,
    Krylov,
    LinearOperator,
    SolveStatus,
    SparseTopology,
    Tolerance,
    analyze,
    empty_factor,
    factor,
    gmres,
    operator_from_jvp,
    operator_from_values,
    refactor,
    solve,
    solve_checked,
)

N = 12
TIGHT = Tolerance(**{"rtol": 1e-12, "atol": 0.0}, max_iter=60, restart=20)
# measured at most 5.1e-15 relative on the GMRES, operator and refactor solutions below, CPU VM
# 2026-10-22, pytest tests/contracts/test_linear_krylov.py
CLOSE = {"rtol": 1e-12, "atol": 1e-13}
# measured 1.8e-10 relative (solutions and gradients) on the lagged preconditioner at rtol
# 1e-10, CPU VM 2026-10-22,
# pytest tests/contracts/test_linear_krylov.py::test_a_lagged_preconditioner_converges_in_the_weighted_norm_and_differentiates_exactly
LAGGED = {"rtol": 2e-9, "atol": 1e-11}


def _random(n=N, seed=0, shift=4.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n)) / np.sqrt(n) + shift * np.eye(n), rng.normal(size=n)


def _operator(a):
    m = jnp.asarray(a)
    return LinearOperator(lambda x: x @ m.T, lambda x: x @ m, shape=a.shape, dtype=m.dtype)


def _wrms(weights, vector):
    return float(np.sqrt(np.mean((np.asarray(weights) * np.asarray(vector)) ** 2)))


def _sparse(n=N, seed=3, dirichlet=False):
    """A banded nonsymmetric pattern with long-range entries and a duplicate."""
    pairs = [(i, j) for i in range(n) for j in (i - 2, i - 1, i, i + 1) if 0 <= j < n]
    pairs += [(0, n - 1), (n - 1, 2), (3, 7), (5, 5)]
    rows, cols = (np.asarray(axis, np.int32) for axis in zip(*pairs))
    values = np.random.default_rng(seed).uniform(-1.0, 1.0, rows.size) + np.where(
        rows == cols, 5.0, 0.0
    )
    bc = np.asarray((1, n - 3), np.int32) if dirichlet else None
    return SparseTopology(rows=rows, cols=cols, n_dofs=n, bc_indices=bc), jnp.asarray(values)


def _assembled(topology, values):
    a = np.zeros((topology.n_dofs, topology.n_dofs))
    np.add.at(a, (topology.rows, topology.cols), np.asarray(values))
    if topology.bc_indices is not None:
        a[topology.bc_indices] = 0.0
        a[topology.bc_indices, topology.bc_indices] = 1.0
    return a


def test_gmres_converges_unpreconditioned_jacobi_and_in_one_exact_iteration():
    """Unpreconditioned GMRES meets a loose tolerance under jit, Jacobi a tight one for a block
    of independent rows, and the exact inverse converges in one iteration whatever error
    weights spread over two to eight decades."""
    a, b = _random()
    exact = np.linalg.solve(a, b)
    x, report = jax.jit(
        lambda r: gmres(
            _operator(a), r, tol=Tolerance(**{"rtol": 1e-6, "atol": 0.0}, max_iter=200, restart=10)
        )
    )(b)
    assert int(report.status) == SolveStatus.SUCCESS
    # measured 2.7e-5 componentwise relative at rtol 1e-6, CPU VM 2026-10-22,
    # pytest tests/contracts/test_linear_krylov.py::test_gmres_converges_unpreconditioned_jacobi_and_in_one_exact_iteration
    np.testing.assert_allclose(x, exact, rtol=1e-3)
    block = np.stack([b, 2.0 * b + 1.0, np.cos(np.arange(N))])
    diagonal = jnp.asarray(np.diag(a))
    x_b, report_b = gmres(_operator(a), block, precondition=lambda v: v / diagonal, tol=TIGHT)
    np.testing.assert_array_equal(report_b.status, [SolveStatus.SUCCESS] * 3)
    np.testing.assert_allclose(x_b, np.linalg.solve(a, block.T).T, **CLOSE)
    inverse = jnp.asarray(np.linalg.inv(a))
    for decades in (2, 4, 8):
        weights = np.logspace(0.0, -decades, N)[np.random.default_rng(1).permutation(N)]
        x_e, report_e = gmres(
            _operator(a), b, precondition=lambda v: inverse @ v, tol=TIGHT, weights=weights
        )
        assert (int(report_e.status), int(report_e.iterations)) == (SolveStatus.SUCCESS, 1)
        np.testing.assert_allclose(x_e, exact, **CLOSE)


def test_the_stopping_rule_is_weighted_and_an_initial_guess_is_honoured():
    """Where the unweighted solve stops, a heavy weight on the worst unknown rejects that
    solution and the weighted solve stops at a solution whose weighted residual, reported and
    recomputed, meets the tolerance; a guess equal to the solution stops at once."""
    a, b = _random(shift=1.5, seed=4)
    tol = Tolerance(**{"rtol": 1e-4, "atol": 0.0}, max_iter=200, restart=40)
    x_p, _ = gmres(_operator(a), b, tol=tol)
    weights = np.ones(N)
    weights[np.argmax(np.abs(b - a @ np.asarray(x_p)) / np.abs(b))] = 1e4
    assert _wrms(weights, b - a @ np.asarray(x_p)) > tol.rtol * _wrms(weights, b)
    x_w, weighted = gmres(_operator(a), b, tol=tol, weights=weights)
    residual = _wrms(weights, b - a @ np.asarray(x_w))
    # measured 2.8e-15 relative between the reported and NumPy's weighted residual, CPU VM
    # 2026-10-22, pytest tests/contracts/test_linear_krylov.py::test_the_stopping_rule_is_weighted_and_an_initial_guess_is_honoured
    np.testing.assert_allclose(float(weighted.residual), residual, rtol=1e-12)
    assert residual <= tol.rtol * _wrms(weights, b)
    _, at_solution = gmres(_operator(a), b, tol=tol, x0=np.linalg.solve(a, b))
    assert (int(at_solution.iterations), int(at_solution.applications)) == (0, 1)


def test_every_gmres_and_krylov_status_is_reachable():
    """MAX_ITERATIONS, STAGNATION, BREAKDOWN, NONFINITE and PRECONDITIONER_FAILURE from GMRES,
    and EMPTY and SINGULAR from a Krylov factor's preconditioner; failures give no finite
    entry, a solve stopped short a finite iterate."""
    a, b = _random()
    singular = a.copy()
    singular[:, 0] = 0.0
    x, report = gmres(_operator(singular), b, tol=Tolerance(1e-12, 0.0, 3, 2))
    assert int(report.status) == SolveStatus.MAX_ITERATIONS and bool(jnp.all(jnp.isfinite(x)))
    rotation = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    _, report = gmres(_operator(rotation), np.asarray([1.0, 0.0]), tol=Tolerance(1e-8, 0, 50, 1))
    assert int(report.status) == SolveStatus.STAGNATION
    _, report = gmres(_operator(np.diag([1.0, 0.0])), np.asarray([0.0, 1.0]), tol=TIGHT)
    assert int(report.status) == SolveStatus.BREAKDOWN
    for rhs, precondition, status in (
        (np.where(np.arange(N) == 3, np.nan, b), None, SolveStatus.NONFINITE),
        (b, lambda v: v * jnp.nan, SolveStatus.PRECONDITIONER_FAILURE),
    ):
        x, report = gmres(_operator(a), rhs, precondition=precondition, tol=TIGHT)
        assert int(report.status) == status and not np.isfinite(np.asarray(x)).any()
    topology, values = _sparse()
    analysis = analyze(topology, backend=Krylov(Dense(), tol=TIGHT))
    broken = jnp.where(jnp.asarray(topology.rows == 4), 0.0, values)
    for held, status in (
        (empty_factor(analysis), SolveStatus.EMPTY),
        (factor(analysis, broken), SolveStatus.SINGULAR),
    ):
        x, _, report = solve_checked(held, jnp.ones(N))
        assert int(report.status) == status and not np.isfinite(np.asarray(x)).any()


def test_operators_apply_the_factored_operator_or_a_jacobian_and_refuse_what_they_lack():
    """``operator_from_values`` applies ``A_bc`` and its transpose; ``operator_from_jvp`` is the
    Jacobian and its transpose, which GMRES solves matrix-free; an operator without a
    transposed action, an unknown tag and contradictory tags are refused."""
    topology, values = _sparse(dirichlet=True)
    a, x = _assembled(topology, values), jnp.asarray(np.cos(np.arange(N)))
    op = operator_from_values(analyze(topology, backend=Dense()), values)
    assert op.tags == frozenset({"square", "nonsymmetric"})
    np.testing.assert_allclose(
        [op.mv(x), op.rmv(x)], [a @ np.asarray(x), a.T @ np.asarray(x)], **CLOSE
    )
    m = jnp.asarray(_random(seed=7)[0])

    def residual(y):
        return m @ y + 0.1 * jnp.sin(y) ** 2

    point = jnp.linspace(-1.0, 1.0, N)
    jacobian = np.asarray(jax.jacfwd(residual)(point))
    matrix_free = operator_from_jvp(residual, point)
    for transpose, oriented in ((False, jacobian), (True, jacobian.T)):
        y, report = jax.jit(
            lambda r, t=transpose: gmres(
                operator_from_jvp(residual, point), r, tol=TIGHT, transpose=t
            )
        )(x)
        assert int(report.status) == SolveStatus.SUCCESS
        np.testing.assert_allclose(y, np.linalg.solve(oriented, x), **CLOSE)
    forward_only = LinearOperator(matrix_free.mv, None, shape=(N, N), dtype=jnp.float64)
    with pytest.raises(FactorContractError, match="no transposed action"):
        forward_only.apply(x, transpose=True)
    for tags, match in ((("banded",), "unknown"), (("spd", "nonsymmetric"), "both")):
        with pytest.raises(FactorContractError, match=match):
            LinearOperator(matrix_free.mv, None, shape=(N, N), dtype=jnp.float64, tags=tags)


class _NoDefaultTolerance(Dense):
    backend_name = "inexact_dense_without_default"
    is_exact = False
    default_tolerance = None


def test_a_krylov_backend_is_inexact_keeps_new_values_on_refactor_and_refuses_bad_configuration():
    """A Krylov factor is marked inexact and a refactor with a false predicate takes the new
    values and their tangent as the operator (keeping the preconditioner); an inexact backend
    without a default tolerance needs ``tol=``; Krylov refuses an inexact inner backend, a zero
    lagged cap, a zero restart and the fp32 rung."""
    topology, values = _sparse()
    analysis = analyze(topology, backend=Krylov(Dense(), tol=TIGHT))
    held = factor(analysis, 1.05 * values)
    assert held.exact is False and factor(analyze(topology, backend=Dense()), values).exact is True
    direction = jnp.asarray(np.random.default_rng(1).uniform(-1, 1, values.shape))
    kept, tangent = jax.jvp(
        lambda v: refactor(held, v, jnp.asarray(False)).values, (values,), (direction,)
    )
    np.testing.assert_array_equal(kept, values)
    np.testing.assert_array_equal(tangent, direction)
    np.testing.assert_allclose(
        solve(refactor(held, values, False), jnp.ones(N))[0],
        np.linalg.solve(_assembled(topology, values), np.ones(N)),
        **CLOSE,
    )
    bare = factor(analyze(topology, backend=_NoDefaultTolerance()), values)
    with pytest.raises(FactorContractError, match="default_tolerance is None.*pass tol="):
        solve(bare, jnp.ones(N))
    for build, match in (
        (lambda: Krylov(Krylov(Dense())), "exact inner backend"),
        (lambda: Krylov(Dense(), lagged_max_iter=0), "lagged_max_iter"),
        (lambda: Krylov(Dense(), tol=Tolerance(1e-8, 0.0, 10, 0)), "Tolerance"),
        (lambda: analyze(topology, backend=Krylov(Dense()), precision="fp32"), "precision"),
    ):
        with pytest.raises(ds.ConfigurationError, match=match):
            build()


def test_a_lagged_preconditioner_converges_in_the_weighted_norm_and_differentiates_exactly():
    """A Krylov factor built 1e-2 away and refactored to the current values with a false
    predicate converges in both directions in two or more iterations, reports NumPy's weighted
    residual below the tolerance, and its solution and its derivatives (weights and initial
    guess carry none) equal the exact Dense backend's."""
    topology, values = _sparse()
    analysis = analyze(topology, backend=Krylov(Dense(), tol=TIGHT))
    a, b = _assembled(topology, values), jnp.linspace(-1.0, 1.0, N)
    weights, guess = jnp.linspace(0.5, 2.0, N), jnp.cos(jnp.arange(N, dtype=jnp.float64))
    stale = values * (
        1.0 + 1e-2 * jnp.asarray(np.random.default_rng(13).uniform(-1, 1, values.shape))
    )
    lagged = Tolerance(**{"rtol": 1e-10, "atol": 0.0}, max_iter=40, restart=20)

    def solved(v, r, transpose=False):
        f = refactor(factor(analysis, stale), v, jnp.asarray(False))
        return solve_checked(f, r, transpose=transpose, tol=lagged, weights=weights, x0=guess)

    for transpose, matrix in ((False, a), (True, a.T)):
        x, _, report = jax.jit(lambda v, r, t=transpose: solved(v, r, t))(values, b)
        assert bool(report.converged) and int(report.iterations) >= 2
        # measured 4.0e-17 absolute between the reported and NumPy's weighted residual, CPU VM
        # 2026-10-22, pytest tests/contracts/test_linear_krylov.py::test_a_lagged_preconditioner_converges_in_the_weighted_norm_and_differentiates_exactly
        recomputed = _wrms(weights, b - matrix @ np.asarray(x))
        np.testing.assert_allclose(float(report.residual), recomputed, rtol=0.0, atol=1e-15)
        assert float(report.residual) <= lagged.rtol * _wrms(weights, b)
        np.testing.assert_allclose(x, np.linalg.solve(matrix, b), **LAGGED)
    exact = analyze(topology, backend=Dense())

    def loss(v, r):
        return jnp.sum(jnp.sin(solved(v, r)[0]))

    def loss_exact(v, r):
        return jnp.sum(jnp.sin(solve(factor(exact, v), r)[0]))

    for got, want in zip(
        jax.jit(jax.grad(loss, argnums=(0, 1)))(values, b),
        jax.grad(loss_exact, argnums=(0, 1))(values, b),
    ):
        np.testing.assert_allclose(got, want, **LAGGED)
