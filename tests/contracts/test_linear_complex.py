"""Linear contract, complex values: every exact backend against the dense reference on a random
sparse complex system and on a battery model's impedance operator, the plain transpose that
reverse mode needs, value lanes, and first derivatives against central differences."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diff_soup.linear import (
    KLU,
    Dense,
    SolveContext,
    SolveStatus,
    SparseTopology,
    analyze,
    empty_factor,
    factor,
    refactor,
    refine,
    solve,
    solve_checked,
)

BACKENDS = [
    pytest.param(Dense, id="dense"),
    pytest.param(KLU, id="klu", marks=pytest.mark.requires_klu),
]
N, BC_ROW = 24, 3
# measured at most 2.7e-16 relative against NumPy over every random-system solve below, CPU
# workstation 2026-11-05, pytest tests/contracts/test_linear_complex.py
_EXACT = 1e-13
# measured at most 1.1e-12 relative between KLU and the dense reference on the battery operator,
# whose condition number reaches 3.4e7, CPU workstation 2026-11-05, same file
_BATTERY = 1e-10
# measured at most 9.5e-10 relative between a derivative and its central difference (step 1e-6),
# CPU workstation 2026-11-05, same file
_FD = 1e-7
#: Dimensionless angular frequencies of the battery operator ``i w M - J``.
OMEGAS = (1e-3, 1.0, 1e3)


def _random_system(seed=0):
    """``(topology, values, matrix)``: a sparse complex system with one Dirichlet row."""
    rng = np.random.default_rng(seed)
    mask = rng.random((N, N)) < 0.15
    np.fill_diagonal(mask, True)
    rows, cols = (axis.astype(np.int32) for axis in np.nonzero(mask))
    values = rng.normal(size=rows.size) + 1j * rng.normal(size=rows.size)
    values = values + np.where(rows == cols, 6.0 - 2.0j, 0.0)
    matrix = np.zeros((N, N), complex)
    np.add.at(matrix, (rows, cols), values)
    matrix[BC_ROW] = 0.0
    matrix[BC_ROW, BC_ROW] = 1.0
    topology = SparseTopology(rows, cols, N, bc_indices=np.asarray([BC_ROW]))
    return topology, jnp.asarray(values), matrix


def _battery():
    """``(topology, jacobian, mass)`` of a porous-electrode lithium-ion cell at rest."""
    path = Path(__file__).resolve().parents[1] / "support" / "battery_jacobian.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    topology = SparseTopology(np.asarray(data["rows"]), np.asarray(data["cols"]), data["n"])
    return topology, np.asarray(data["jacobian"]), np.asarray(data["mass"])


def _rhs(n, seed=1, shape=()):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(*shape, n)) + 1j * rng.normal(size=(*shape, n)))


def _relative(got, want):
    got, want = np.asarray(got), np.asarray(want)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def _context(make):
    return None if make.factor_residency == "arrays" else SolveContext()


def _solver(make, topology, ctx):
    """``x(values, b, transpose)``: a factor of ``values`` then one solve."""
    analysis = analyze(topology, backend=make())

    def run(values, b, transpose=False):
        return solve(factor(analysis, values, ctx=ctx), b, transpose=transpose)[0]

    return run


@pytest.mark.parametrize("transpose", [False, True], ids=["plain", "transposed"])
@pytest.mark.parametrize("make", BACKENDS)
def test_complex_solves_match_numpy_and_the_dense_reference(make, transpose):
    """A complex factor solves one and several right-hand sides, plainly and transposed, as
    NumPy does with ``A`` and ``A.T``, and as the dense reference does on the battery operator
    at three frequencies."""
    topology, values, matrix = _random_system()
    ctx = _context(make)
    run = _solver(make, topology, ctx)
    b, block = _rhs(N), _rhs(N, seed=2, shape=(3,))
    op = matrix.T if transpose else matrix
    x = run(values, b, transpose)
    assert x.dtype == jnp.complex128
    assert _relative(x, np.linalg.solve(op, b)) <= _EXACT
    assert _relative(run(values, block, transpose), np.linalg.solve(op, block.T).T) <= _EXACT
    battery, jacobian, mass = _battery()
    reference = _solver(Dense, battery, None)
    sparse = _solver(make, battery, ctx)
    rhs = _rhs(battery.n_dofs, seed=3)
    for omega in OMEGAS:
        v = jnp.asarray(1j * omega * mass - jacobian)
        assert _relative(sparse(v, rhs, transpose), reference(v, rhs, transpose)) <= _BATTERY


@pytest.mark.parametrize("make", BACKENDS)
def test_transpose_is_the_plain_transpose_that_linear_transpose_derives(make):
    """``jax.linear_transpose`` of a complex solve is the solve with ``transpose=True``, which is
    ``A^T``, not the conjugate transpose ``A^H``; a factor of orientation ``"AT"`` solves
    ``A^T`` too."""
    topology, values, matrix = _random_system()
    ctx = _context(make)
    analysis = analyze(topology, backend=make())
    f = factor(analysis, values, ctx=ctx)
    b = _rhs(N)
    (pulled,) = jax.linear_transpose(lambda rhs: solve(f, rhs)[0], b)(b)
    transposed, _ = solve(f, b, transpose=True)
    np.testing.assert_array_equal(pulled, transposed)
    assert _relative(transposed, np.linalg.solve(matrix.T, b)) <= _EXACT
    # measured 0.63 relative between the A^T and the A^H solution of this system, CPU
    # workstation 2026-11-05, pytest tests/contracts/test_linear_complex.py
    assert _relative(transposed, np.linalg.solve(matrix.conj().T, b)) > 0.1
    oriented, _ = solve(factor(analysis, values, ctx=ctx, orientation="AT"), b)
    assert _relative(oriented, np.linalg.solve(matrix.T, b)) <= _EXACT


@pytest.mark.parametrize("make", BACKENDS)
def test_a_real_factor_solves_a_complex_right_hand_side(make):
    """A complex right-hand side on a real factor has the complex solution of its real and
    imaginary parts, plainly and transposed; a real one stays real."""
    topology, values, matrix = _random_system()
    run = _solver(make, topology, _context(make))
    real = jnp.real(values)
    real_matrix = matrix.real.copy()
    real_matrix[BC_ROW, BC_ROW] = 1.0
    b = _rhs(N, shape=(2,))
    for transpose, op in ((False, real_matrix), (True, real_matrix.T)):
        x = run(real, b, transpose)
        assert x.dtype == jnp.complex128
        assert _relative(x, np.linalg.solve(op, b.T).T) <= _EXACT
    assert run(real, jnp.real(b[0])).dtype == jnp.float64


@pytest.mark.parametrize("make", BACKENDS)
def test_complex_value_lanes_solve_each_matrix(make):
    """``jax.vmap`` over complex values, under ``jit``, solves each lane's own matrix: the
    battery operator at three frequencies."""
    topology, jacobian, mass = _battery()
    ctx = _context(make)
    run = _solver(make, topology, ctx)
    lanes = jnp.asarray([1j * omega * mass - jacobian for omega in OMEGAS])
    b = _rhs(topology.n_dofs, seed=4)
    xs = jax.jit(jax.vmap(lambda v: run(v, b)))(lanes)
    reference = _solver(Dense, topology, None)
    for lane, v in enumerate(lanes):
        assert _relative(xs[lane], reference(v, b)) <= _BATTERY


@pytest.mark.parametrize("make", BACKENDS)
def test_an_empty_complex_factor_refactors_in_a_loop_and_reports(make):
    """An empty complex factor solves to ``nan`` with ``EMPTY``; refactored in a ``lax.scan``
    over three matrices it solves each, ``solve_checked`` reports success with a small
    residual and ``refine`` keeps the solution."""
    topology, values, matrix = _random_system()
    ctx = _context(make)
    analysis = analyze(topology, backend=make())
    empty = empty_factor(analysis, ctx=ctx, dtype=jnp.complex128)
    b = _rhs(N)
    x, _, report = solve_checked(empty, b)
    assert int(report.status) == SolveStatus.EMPTY and np.all(np.isnan(np.asarray(x)))
    scales = jnp.asarray([1.0, 2.0 + 1.0j, -0.5j])

    def step(f, scale):
        f = refactor(f, scale * values, True)
        x, f, report = solve_checked(f, b)
        return f, (x, report.status, report.residual)

    f, (xs, status, residual) = jax.jit(lambda f: jax.lax.scan(step, f, scales))(empty)
    assert np.all(np.asarray(status) == SolveStatus.SUCCESS)
    # measured at most 1.3e-15 for the largest |b - M x| entry, CPU workstation 2026-11-05,
    # pytest tests/contracts/test_linear_complex.py
    assert float(jnp.max(residual)) <= 1e-13
    for lane, scale in enumerate(np.asarray(scales)):
        assert _relative(xs[lane], np.linalg.solve(scale * matrix + _unit(scale), b)) <= _EXACT
    refined, _, _ = refine(f, b, values=scales[-1] * values)
    assert _relative(refined, xs[-1]) <= _EXACT


def _unit(scale):
    """The Dirichlet row stays a unit row whatever the scale of the values."""
    unit = np.zeros((N, N), complex)
    unit[BC_ROW, BC_ROW] = 1.0 - scale
    return unit


def _central(fun, args, directions, step=1e-6):
    """The central difference of ``fun`` at ``args`` along ``directions``."""
    plus = fun(*(a + step * d for a, d in zip(args, directions)))
    minus = fun(*(a - step * d for a, d in zip(args, directions)))
    return (plus - minus) / (2 * step)


def _case(name, values, b):
    """``(inputs, primals)``: the values and right-hand side as a function of the primals.

    ``complex``: complex values and right-hand side. ``real``: their real parts. ``parameter``:
    complex values and right-hand side driven by two real parameters, so a complex factor is
    differentiated along a real direction.
    """
    if name == "real":
        return (lambda v, rhs: (v, rhs)), (jnp.real(values), jnp.real(b))
    if name == "complex":
        return (lambda v, rhs: (v, rhs)), (values, b)
    dv, db = _rhs(values.size, seed=7), _rhs(b.size, seed=8)
    return (lambda p: (values + p[0] * dv, b * jnp.exp(1j * p[1]) + p[0] * db)), (
        jnp.asarray([0.3, 0.2]),
    )


def _direction(arg, seed):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=arg.shape)
    if jnp.iscomplexobj(arg):
        d = d + 1j * rng.normal(size=arg.shape)
    return jnp.asarray(d)


@pytest.mark.parametrize("case", ["complex", "real", "parameter"])
@pytest.mark.parametrize("make", BACKENDS)
def test_jvp_and_vjp_of_a_factor_then_solve_match_central_differences(make, case):
    """``jax.jvp`` and ``jax.vjp`` through ``factor`` then ``solve`` match central differences:
    complex values and right-hand side, real ones, and a complex factor along real parameters,
    for which the cotangent is the real part of ``ct^T J``."""
    topology, values, _ = _random_system()
    run = _solver(make, topology, _context(make))
    inputs, primals = _case(case, values, _rhs(N))

    def fun(*args):
        return run(*inputs(*args))

    directions = tuple(_direction(p, seed) for seed, p in enumerate(primals))
    expected = _central(fun, primals, directions)
    _, tangent = jax.jvp(fun, primals, directions)
    assert _relative(tangent, expected) <= _FD
    out, pullback = jax.vjp(fun, *primals)
    cotangent = _rhs(N, seed=9) if jnp.iscomplexobj(out) else jnp.real(_rhs(N, seed=9))
    paired = sum(jnp.sum(p * d) for p, d in zip(pullback(cotangent), directions))
    want = jnp.sum(cotangent * expected)
    want = jnp.real(want) if case == "parameter" else want
    assert abs(complex(paired) - complex(want)) <= _FD * abs(complex(want))
