"""Linear contract, host backends (KLU and SuperLU_MT), whose factors live in native slots of
a SolveContext: only what differs from the dense reference is proven here, their settings, and
that a solve by name keeps its backend."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.linear import (
    KLU,
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
    solve,
    solve_checked,
)
from tests.support.problems import reaction_diffusion_recipe

BACKENDS = {"klu": KLU, "superlu": SuperLU}
MARKS = {"klu": pytest.mark.requires_klu, "superlu": pytest.mark.requires_superlu}
HOSTS = [pytest.param(name, id=name, marks=MARKS[name]) for name in BACKENDS]
SETTINGS = {
    "klu": {"ordering": "colamd", "btf": False},
    "superlu": {"ordering": "natural", "relax": 4},
}
N, BC_ROW = 6, 2
# measured at most 5.3e-15 componentwise relative against NumPy over every system below (1.0e-14
# on the 1000-row arrowhead), CPU VM 2026-10-22, pytest tests/contracts/test_linear_host.py; 0.0
# between the shared-storage tangent program and its own-storage twin (KLU), arm64 Mac 2026-10-23;
# at most 4.7e-16 for the configured backends and 0.0 between a host projection and the dense
# one, L40S node (CPU device) 2026-10-26, same file
EXACT = {"rtol": 1e-12, "atol": 1e-13}


def _pattern():
    """A 6x6 tridiagonal pattern plus duplicates and long-range entries, one Dirichlet row."""
    rows = [*range(N), *range(N - 1), *range(1, N), 3, 0, 2, 0, 4, 2]
    cols = [*range(N), *range(1, N), *range(N - 1), 3, 1, 2, 5, 1, 4]
    return SparseTopology(np.asarray(rows), np.asarray(cols), N, bc_indices=np.asarray([BC_ROW]))


def _values(topology, seed=0):
    values = np.random.default_rng(seed).uniform(-1.0, 1.0, topology.rows.size)
    return values + 5.0 * (topology.rows == topology.cols)


def _matrix(topology, values):
    a = np.zeros((N, N))
    np.add.at(a, (topology.rows, topology.cols), values)
    a[BC_ROW] = 0.0
    a[BC_ROW, BC_ROW] = 1.0
    return a


_B = np.linspace(-1.0, 2.0, N)


def _delta(backend, run):
    """The change of the backend's native counters over ``run()``."""
    before = backend.native_stats()
    jax.block_until_ready(run())
    after = backend.native_stats()
    return {key: after[key] - before[key] for key in after}


@pytest.mark.parametrize(
    "make,setting,refusal",
    [
        (KLU, {"ordering": "metis"}, "KLU option ordering"),
        (KLU, {"pivot_tolerance": 0.0}, "KLU option pivot_tolerance"),
        (SuperLU, {"ordering": "amd"}, "SuperLU option ordering"),
        (SuperLU, {"panel_size": -1}, "SuperLU option panel_size"),
    ],
)
def test_host_settings_are_content_hashed_and_refused_when_the_backend_is_built(
    make, setting, refusal
):
    """A host backend without settings equals and hashes as another, a setting makes it differ,
    and a setting analyze would refuse raises when the backend is built, naming the setting."""
    assert make() == make() and hash(make()) == hash(make())
    assert make(ordering="natural" if make is SuperLU else "colamd") != make()
    with pytest.raises(ds.ConfigurationError, match=refusal):
        make(**setting)


@pytest.mark.parametrize("name", HOSTS)
def test_host_settings_reach_the_native_analysis_and_an_explicit_option_overrides_one(name):
    """A backend without settings analyzes with no options and gets the memoised analysis of
    another; a backend's settings are the options its native analysis is built with, a key of
    an explicit ``options=`` overrides the setting, and the configured factor solves exactly."""
    make, settings, topology = BACKENDS[name], SETTINGS[name], _pattern()
    default = analyze(topology, backend=make())
    assert default.options == () and analyze(topology, backend=make()) is default
    configured = analyze(topology, backend=make(**settings))
    assert configured.options == tuple(sorted(settings.items()))
    assert configured.backend_analysis.ordering == settings["ordering"]
    ordering = {"ordering": default.backend_analysis.ordering}
    overridden = analyze(topology, backend=make(**settings), options=ordering)
    assert overridden.backend_analysis.ordering == ordering["ordering"]
    with SolveContext() as ctx:
        x = solve(factor(configured, _values(topology), ctx=ctx), _B)[0]
    np.testing.assert_allclose(
        x, np.linalg.solve(_matrix(topology, _values(topology)), _B), **EXACT
    )


@pytest.mark.parametrize("name", HOSTS)
def test_host_solves_match_numpy_and_the_dense_reference(name):
    """Both orientations and a block of right-hand sides match NumPy and the dense backend."""
    topology, values = _pattern(), _values(_pattern())
    a, block = _matrix(topology, values), np.stack([_B, 2.0 * _B + 1.0])
    dense = factor(analyze(topology, backend=Dense()), values)
    with SolveContext() as ctx:
        f = jax.jit(lambda v: factor(analyze(topology, backend=BACKENDS[name]()), v, ctx=ctx))(
            values
        )
        for transpose, matrix in ((False, a), (True, a.T)):
            x = solve(f, _B, transpose=transpose)[0]
            np.testing.assert_allclose(x, np.linalg.solve(matrix, _B), **EXACT)
            np.testing.assert_allclose(x, solve(dense, _B, transpose=transpose)[0], **EXACT)
        np.testing.assert_allclose(solve(f, block)[0], np.linalg.solve(a, block.T).T, **EXACT)


@pytest.mark.requires_klu
@pytest.mark.requires_superlu
def test_each_host_library_recovers_where_its_defaults_fail():
    """A KLU refactor whose kept pivot sequence meets a zero pivot factors afresh, counts the
    fallback and solves exactly; SuperLU_MT factors an arrowhead whose natural-order LU fills
    far beyond the library's default allowance of 50 nnz(A)."""
    analysis = analyze(SparseTopology.full(2), backend=KLU())
    with SolveContext() as ctx:
        run = jax.jit(lambda v: solve(factor(analysis, v, ctx=ctx), np.asarray([1.0, 2.0]))[0])
        run(np.asarray([2.0, 1.0, 1.0, 2.0]))
        counts = _delta(KLU(), lambda: run(np.asarray([4.0, 2.0, 2.0, 4.0])))
        assert (counts["refactorizations"], counts["refactor_fallbacks"]) == (1, 0)
        permuted = np.asarray([0.0, 1.0, 1.0, 0.0])
        counts = _delta(KLU(), lambda: run(permuted))
        assert (counts["factorizations"], counts["refactor_fallbacks"]) == (1, 1)
        np.testing.assert_allclose(run(permuted), [2.0, 1.0], **EXACT)
    n, idx = 1000, np.arange(1000)
    rows = np.concatenate([idx, np.zeros(n - 1, int), idx[1:]])
    cols = np.concatenate([idx, idx[1:], np.zeros(n - 1, int)])
    values = np.concatenate([np.full(n, 4.0 * n), np.ones(n - 1), np.ones(n - 1)])
    analysis = analyze(
        SparseTopology(rows, cols, n), backend=SuperLU(), options={"ordering": "natural"}
    )
    b = np.linspace(-1.0, 1.0, n)
    with SolveContext() as ctx:
        x, _, report = solve_checked(factor(analysis, values, ctx=ctx), b)
    a = np.zeros((n, n))
    np.add.at(a, (rows, cols), values)
    assert int(report.status) == SolveStatus.SUCCESS
    np.testing.assert_allclose(x, np.linalg.solve(a, b), **EXACT)


@pytest.mark.parametrize("name", HOSTS)
def test_the_order_word_makes_a_refactor_wait_for_the_solve_before_it(name):
    """In factor, solve, in-place refactor, solve, the refactor's order operand is the order
    word the first solve returned: the solve reads its own factor, nothing is rebuilt."""
    topology = _pattern()
    values, other = _values(topology), _values(topology, seed=6)
    analysis = analyze(topology, backend=BACKENDS[name]())
    with SolveContext() as ctx:

        def program(v, w):
            x1, f = solve(factor(analysis, v, ctx=ctx), _B)
            return x1, solve(refactor(f, w, True, adopt=True), _B)[0]

        eqns = jax.make_jaxpr(program)(values, other).jaxpr.eqns
        first_solve = next(e for e in eqns if e.primitive.name == "phased_solve")
        refactored = next(e for e in eqns if e.primitive.name == "phased_refactor")
        assert refactored.invars[-1] is first_solve.outvars[-1]
        xs = []
        assert (
            _delta(BACKENDS[name](), lambda: xs.extend(jax.jit(program)(values, other)))["rebuilds"]
            == 0
        )
        for x, v in zip(xs, (values, other)):
            np.testing.assert_allclose(x, np.linalg.solve(_matrix(topology, v), _B), **EXACT)


@pytest.mark.parametrize("name", HOSTS)
def test_lanes_keep_their_own_slots_and_eager_refactors_share_one(name):
    """A 3 x 3 vmap of one factor call site keeps nine slots, factors nine times and never
    rebuilds, also when run again; eager refactors of one factor write one slot, and a solve of
    an older factor whose slot was overwritten rebuilds it from its own values."""
    topology = _pattern()
    grid = np.stack(
        [np.stack([_values(topology, seed=40 + 3 * j + k) for k in range(3)]) for j in range(3)]
    )
    backend = BACKENDS[name]()
    analysis = analyze(topology, backend=backend)
    with SolveContext() as ctx:
        run = jax.jit(jax.vmap(jax.vmap(lambda v: solve(factor(analysis, v, ctx=ctx), _B)[0])))
        first = _delta(backend, lambda: run(grid))
        assert (
            first["live_slots"],
            first["factorizations"] + first["refactorizations"],
            first["rebuilds"],
        ) == (9, 9, 0)
        again = _delta(backend, lambda: run(2.0 * grid))
        assert (again["live_slots"], again["rebuilds"]) == (0, 0)
        xs = np.asarray(run(grid))
    expected = [[np.linalg.solve(_matrix(topology, v), _B) for v in row] for row in grid]
    np.testing.assert_allclose(xs, expected, **EXACT)
    values, second, third = (_values(topology, seed) for seed in (0, 31, 32))
    backend = BACKENDS[name]()
    with SolveContext() as ctx:
        f = factor(analyze(topology, backend=backend), values, ctx=ctx)
        g = refactor(f, second, True)
        h = refactor(g, third, True)
        assert int(h.handle[1]) == int(g.handle[1]) != int(f.handle[1])
        solutions = {}
        counts = _delta(
            backend,
            lambda: solutions.update(
                {id(v): solve(held, _B)[0] for held, v in ((g, second), (h, third), (f, values))}
            ),
        )
        assert counts["rebuilds"] == 2
        for v in (second, third, values):
            np.testing.assert_allclose(
                solutions[id(v)], np.linalg.solve(_matrix(topology, v), _B), **EXACT
            )


@pytest.mark.parametrize("name", HOSTS)
def test_a_tangent_through_twice_refactored_shared_storage_equals_one_through_its_own(name):
    """Under jit, jax.jvp of refactor, solve, refactor on one shared-storage factor, then a solve
    of the second factor on the first solve's tangent, equals the same program on a factor of
    its own storage, whichever of the older factor's tangent solve and the second refactor runs
    first: a tangent solve that factors the storage again leaves it untrusted for the solve
    after it."""
    topology = _pattern()
    first, second, dfirst, dsecond = (_values(topology, seed) for seed in (50, 51, 52, 53))
    analysis = analyze(topology, backend=BACKENDS[name]())
    with SolveContext() as ctx:

        def program(storage):
            def straight(v, w):
                g = empty_factor(analysis, ctx=ctx, storage=storage)
                x1, g = solve(refactor(g, v, True), _B)
                return x1, refactor(g, w, True)

            def run(v, w, dv, dw):
                (x1, held), (dx1, _) = jax.jvp(straight, (v, w), (dv, dw))
                return x1, dx1, solve(held, dx1)[0]

            return jax.jit(run)(first, second, dfirst, dsecond)

        shared, own = program("shared"), program(None)
    for got, expected in zip(shared, own):
        np.testing.assert_allclose(got, expected, **EXACT)


@pytest.mark.parametrize("name", HOSTS)
def test_native_statistics_count_one_factorization_per_setup_and_release_their_slots(name):
    """A scan that solves with its carried factor and then refactors it in place factors its
    two sites once, refactors once per later step, rebuilds nothing, and the context's
    release frees every slot it took."""
    topology = _pattern()
    steps = np.stack([_values(topology, seed) for seed in range(6, 10)])
    backend = BACKENDS[name]()
    live = backend.native_stats()["live_slots"]
    with SolveContext() as ctx:

        def body(f, v):
            x, f = solve(f, _B)
            return refactor(f, v, True), x

        run = jax.jit(
            lambda v0, vs: jax.lax.scan(
                body, factor(analyze(topology, backend=backend), v0, ctx=ctx), vs
            )[1]
        )
        counts = _delta(backend, lambda: run(_values(topology), steps))
        names = ("factorizations", "refactorizations", "rebuilds")
        assert [counts[k] for k in names] == [2, len(steps) - 1, 0]
    assert backend.native_stats()["live_slots"] == live


@pytest.mark.parametrize("name", HOSTS)
def test_a_host_backend_refuses_a_program_compiled_for_a_gpu(name):
    """Lowering a host factorization for the cuda platform raises FactorContractError naming
    the cpu platform and the cuda target, without a GPU present."""
    analysis = analyze(_pattern(), backend=BACKENDS[name]())
    with SolveContext() as ctx:
        traced = jax.jit(lambda v: solve(factor(analysis, v, ctx=ctx), _B)[0]).trace(
            jnp.asarray(_values(_pattern()))
        )
        with pytest.raises(FactorContractError, match="cpu platform only.*cuda"):
            traced.lower(lowering_platforms=("cuda",))


_DIFFUSION = reaction_diffusion_recipe(n=12)
_LANE_SCALES = jnp.asarray([0.5, 1.0, 1.5])
# measured 0.0 between the KLU lanes and the dense batch, CPU VM 2026-10-22, and between the
# SuperLU lanes and it, L40S node (CPU device) 2026-10-26,
# pytest tests/contracts/test_linear_host.py::test_a_batched_solve_above_ten_unknowns_keeps_the_host_backend_it_names
_LANE_BOUND = 1e-12


def _batched_end_states(linear):
    """A vmapped JIDA solve of a twelve-unknown diffusion over three initial scales: the end
    states and the realized linear backend."""
    problem, y0 = _DIFFUSION.build(), _DIFFUSION.assemble_state({})
    solver = ds.JIDA.stock(linear=linear, **{"rtol": 1e-8, "atol": 1e-10})

    def lane(scale):
        solution = ds.pdesolve(problem, solver, (0.0, 0.5), scale * y0, saveat=ds.SaveAt(t1=True))
        return solution.ys[-1], solution.stats

    ends, stats = jax.vmap(lane)(_LANE_SCALES)
    return ends, stats.linear_backend


@pytest.mark.parametrize(
    "linear,realized",
    [
        pytest.param("klu", "klu", marks=pytest.mark.requires_klu),
        pytest.param("auto", "klu", marks=pytest.mark.requires_klu),
        pytest.param("superlu", "superlu", marks=pytest.mark.requires_superlu),
    ],
)
def test_a_batched_solve_above_ten_unknowns_keeps_the_host_backend_it_names(linear, realized):
    """Under vmap a solve that names KLU or SuperLU, or asks for ``auto`` above ten unknowns on
    CPU (KLU), keeps that backend (its batches run lane by lane) and every lane equals the dense
    batch."""
    ends, backend = _batched_end_states(linear)
    assert backend == realized
    np.testing.assert_allclose(ends, _batched_end_states("dense")[0], rtol=_LANE_BOUND)


_CUBIC = ds.MassProblem(
    rhs=lambda t, y, theta, control: jnp.stack([-theta * y[0], y[1] ** 3 + y[1] - theta * y[0]]),
    mass=ds.MassOperator.sparse(
        [0],
        [0],
        lambda t, y, theta, control: jnp.ones((1,), y.dtype),
        n=2,
        dependencies=ds.Dependencies(time=False, state=False, parameters=False, control=False),
    ),
    structure=ds.DAEStructure(differential_vars=np.array([True, False])),
)


@pytest.mark.parametrize("name", HOSTS)
def test_the_algebraic_projection_takes_a_host_backend_by_name(name):
    """``AlgebraicProjectionOptions(linear_solver=name)`` projects on that host backend to the
    root of ``z^3 + z = theta y0`` the dense projection finds."""

    def root(linear_solver):
        tolerances = {"rtol": 1e-13, "atol": 1e-14}
        options = ds.AlgebraicProjectionOptions(linear_solver=linear_solver, **tolerances)
        state = jnp.asarray([1.3, 0.0])
        projected = ds.project_algebraic(
            _CUBIC, t=0.0, state=state, seed=state, theta=jnp.asarray(0.8), options=options
        )
        return projected.state

    np.testing.assert_allclose(root(name), root("dense"), **EXACT)
