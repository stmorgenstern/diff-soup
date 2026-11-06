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
    CuDSS,
    Dense,
    FactorContractError,
    SolveContext,
    SolveStatus,
    SparseTopology,
    SuperLU,
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
    pytest.param(SuperLU, id="superlu", marks=pytest.mark.requires_superlu),
    pytest.param(CuDSS, id="cudss", marks=pytest.mark.requires_cudss),
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
# measured at most 1.1e-7 relative against NumPy for complex64 cuDSS solves at the fp32 rung,
# RTX 4070 Ti 2026-11-05 (.ports/probe_cudss_complex.py of the complex GPU work)
_FP32 = 1e-4
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
    # cuDSS's default complex solve is not bit-reproducible run to run (1 ulp at one entry of 24
    # over six transposed solves, RTX 4070 Ti, cuDSS 1.0.0.35, 2026-11-05); its deterministic mode is
    analysis = analyze(topology, backend=make(deterministic=True) if make is CuDSS else make())
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
        before = _solves(make)
        x = run(real, b, transpose)
        assert x.dtype == jnp.complex128
        assert _relative(x, np.linalg.solve(op, b.T).T) <= _EXACT
        assert before is None or _solves(make) - before == 1
    assert run(real, jnp.real(b[0])).dtype == jnp.float64


def _solves(make):
    """The backend's native solve count; ``None`` for a backend whose factors are arrays."""
    stats = getattr(make, "native_stats", None) or getattr(make, "contract_stats", None)
    return None if stats is None else stats()["solves"]


@pytest.mark.parametrize("make", BACKENDS)
def test_complex_values_report_non_finite_and_singular_statuses(make):
    """A complex factor with a ``nan`` entry reports ``NONFINITE`` and one of zero values
    ``SINGULAR``, each with a ``nan`` solution."""
    topology, values, _ = _random_system()
    analysis, ctx = analyze(topology, backend=make()), _context(make)
    for bad, status in (
        (values.at[0].set(jnp.nan), SolveStatus.NONFINITE),
        (jnp.zeros_like(values), SolveStatus.SINGULAR),
    ):
        x, _, report = solve_checked(factor(analysis, bad, ctx=ctx), _rhs(N))
        assert int(report.status) == status and np.all(np.isnan(np.asarray(x)))


@pytest.mark.parametrize("make", BACKENDS)
def test_a_real_and_a_complex_factor_on_one_shared_storage_each_solve(make):
    """A real and a complex factor started on one shared storage name and refactored in turn,
    each call its own execution, each solve their own matrix, so the storage switches kind and
    refactors complex values in place."""
    topology, values, matrix = _random_system()
    analysis, ctx = analyze(topology, backend=make()), _context(make)
    real, b = jnp.real(values), _rhs(N)
    real_matrix = matrix.real.copy()
    real_matrix[BC_ROW, BC_ROW] = 1.0
    held = [
        empty_factor(analysis, ctx=ctx, dtype=t, storage="kinds")
        for t in (real.dtype, values.dtype)
    ]

    for scale in (1.0, 2.0):
        for k, (v, op) in enumerate(((real, real_matrix), (values, matrix))):
            held[k] = refactor(held[k], scale * v, True)
            x = solve(held[k], b)[0]
            assert _relative(x, np.linalg.solve(scale * op + _unit(scale), b)) <= _EXACT


@pytest.mark.parametrize("make", BACKENDS)
def test_complex_values_on_an_spd_analysis_are_refused(make):
    """An analysis declared ``spd`` refuses complex values in ``factor`` and ``empty_factor``:
    a complex symmetric matrix is never positive definite, and a Hermitian one is not
    symmetric."""
    topology, values, _ = _random_system()
    analysis = analyze(SparseTopology(topology.rows, topology.cols, N), backend=make(), spd=True)
    ctx = _context(make)
    with pytest.raises(FactorContractError, match="spd=True"):
        factor(analysis, values, ctx=ctx)
    with pytest.raises(FactorContractError, match="spd=True"):
        empty_factor(analysis, ctx=ctx, dtype=jnp.complex128)


@pytest.mark.requires_cudss
@pytest.mark.parametrize("option", [{"downcast_factors": True}, {"emulation_mantissa_bits": 40}])
def test_cudss_refuses_a_float64_precision_option_on_complex_values(option):
    """``downcast_factors`` and ``emulation_mantissa_bits`` act on float64 factors only, so a
    complex factor of an analysis that sets one is refused when it is traced."""
    topology, values, _ = _random_system()
    analysis = analyze(topology, backend=CuDSS(**option))
    with pytest.raises(FactorContractError, match=next(iter(option))):
        factor(analysis, values, ctx=SolveContext())


@pytest.mark.requires_cudss
@pytest.mark.parametrize("transpose", [False, True], ids=["plain", "transposed"])
def test_cudss_real_lanes_solve_a_complex_right_hand_side_as_one_batch(transpose):
    """Real values as ``vmap`` lanes, one native batch, solve each lane's complex right-hand
    side as one real block of its parts."""
    topology, values, matrix = _random_system()
    analysis, ctx = analyze(topology, backend=CuDSS()), SolveContext()
    real_matrix = matrix.real.copy()
    real_matrix[BC_ROW, BC_ROW] = 1.0
    scales, b = jnp.asarray([1.0, 2.0]), _rhs(N, shape=(2,))

    def one(v, rhs):
        return solve(factor(analysis, v, ctx=ctx), rhs, transpose=transpose)[0]

    xs = jax.jit(jax.vmap(one))(scales[:, None] * jnp.real(values), b)
    for x, scale, rhs in zip(xs, np.asarray(scales), b):
        op = scale * real_matrix + _unit(scale).real
        assert _relative(x, np.linalg.solve(op.T if transpose else op, rhs)) <= _EXACT


@pytest.mark.requires_cudss
def test_cudss_complex64_lanes_solve_at_the_fp32_rung():
    """complex64 values as ``vmap`` lanes at the ``"fp32"`` rung, one native batch, solve each
    lane's matrix in complex64 to float32 accuracy."""
    topology, values, matrix = _random_system()
    analysis, ctx = analyze(topology, backend=CuDSS(), precision="fp32"), SolveContext()
    scales, b = jnp.asarray([1.0, 2.0 + 1.0j], jnp.complex64), _rhs(N).astype(jnp.complex64)

    def one(v):
        return solve(factor(analysis, v, ctx=ctx), b)[0]

    xs = jax.jit(jax.vmap(one))(scales[:, None] * values.astype(jnp.complex64))
    assert xs.dtype == jnp.complex64
    for x, scale in zip(xs, np.asarray(scales)):
        assert _relative(x, np.linalg.solve(scale * matrix + _unit(scale), b)) <= _FP32


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


@pytest.mark.requires_klu
@pytest.mark.parametrize("kind", ["real", "complex"])
def test_klu_resident_bytes_count_the_factor_and_fall_to_zero_on_release(kind):
    """A KLU factor of the battery operator holds at least its L and U entries at the element
    size, and at most twice that plus its value copies, KLU's per-row arrays and workspace and
    the off-diagonal entries; its context holds nothing once released."""
    topology, jacobian, mass = _battery()
    values = jnp.asarray((1j if kind == "complex" else 1.0) * 1e-3 * mass - jacobian)
    element, n, nnz = values.dtype.itemsize, topology.n_dofs, topology.rows.size
    before = KLU.native_stats()["live_lu_entries"]
    ctx = SolveContext()
    b = _rhs(n, seed=5) if kind == "complex" else jnp.ones(n)
    jax.block_until_ready(solve(factor(analyze(topology, backend=KLU()), values, ctx=ctx), b))
    entries = KLU.native_stats()["live_lu_entries"] - before
    held = ctx.resident_bytes()
    # An L or U entry and its int32 row index fit in 1.5 elements. The copies are the values and
    # the compressed matrix; the row scaling, permutations, pointers and solve workspace take
    # at most 4 elements and 40 bytes a row; an off-diagonal entry takes an element and an index.
    extra = 2 * nnz * element + n * (4 * element + 40) + nnz * (element + 4)
    assert entries > 0
    assert entries * element <= held <= 2 * entries * element + extra
    ctx.release()
    assert ctx.resident_bytes() == 0
    assert KLU.native_stats()["live_lu_entries"] == before


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
