"""Native backends through the public solvers: KLU projected outputs and events, held-factor
gradients against the dense reference under replay, the Diffrax stepper, and the cuDSS
engines' setups and replays (owed on a CPU host)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.domains import cartesian_grid_1d
from diff_soup.linear import SparseTopology
from diff_soup.timestepping import Bisection
from tests.support.problems import ConstrainedDAE, DecayODE, SparseChainDAE

#: Solver settings (requested tolerances, not asserted bounds).
_FINE = {"rtol": 1e-8, "atol": 1e-10}
_ROOT = Bisection(**{"rtol": 1e-10, "atol": 1e-12})
_PROJECTED = ds.DenseOutputPolicy(mode="projected")
_CHAIN, _DECAY, _THETA = SparseChainDAE(n=8), DecayODE(), jnp.asarray(0.4)
# measured 1.9e-8 relative to the exact samples, 1.9e-16 on the algebraic rows and 1.1e-7
# relative to the exact tangent (5.4e-14 from the dense route's), CPU VM 2026-10-22,
# pytest tests/integration/test_native_backends.py::test_klu_projected_samples_lie_on_the_manifold_and_differentiate_exactly
_SAMPLE, _MANIFOLD, _TANGENT = 1e-6, 1e-12, 1e-6


@pytest.mark.requires_klu
def test_klu_projected_samples_lie_on_the_manifold_and_differentiate_exactly():
    """Projected samples of a sparse DAE on KLU equal the exact solution, satisfy the
    algebraic rows, and carry the exact tangent."""
    ts = jnp.asarray([0.1, 0.25, 0.4])
    saveat = ds.SaveAt(ts=ts, output=_PROJECTED)

    def samples(theta, linear):
        solution = ds.pdesolve(
            _CHAIN.problem,
            ds.JNDF.stock(linear=linear, **_FINE),
            (0.0, 0.5),
            _CHAIN.initial_state(),
            theta=theta,
            saveat=saveat,
            consistent_ic=True,
        )
        return jnp.concatenate([solution.ys, solution.lambdas], axis=1)

    sampled, tangent = jax.jvp(lambda th: samples(th, "klu"), (_THETA,), (1.0,))
    exact = jnp.stack([_CHAIN.analytical(t, _THETA) for t in ts])
    np.testing.assert_allclose(sampled, exact, rtol=0.0, atol=_SAMPLE * float(jnp.max(exact)))
    rows = jnp.stack([_CHAIN.problem.rhs(t, y, _THETA, None) for t, y in zip(ts, sampled)])
    np.testing.assert_allclose(rows[:, _CHAIN.n :], 0.0, atol=_MANIFOLD)
    exact_tangent = jax.jvp(
        lambda th: jnp.stack([_CHAIN.analytical(t, th) for t in ts]), (_THETA,), (1.0,)
    )[1]
    scale = float(jnp.max(jnp.abs(exact_tangent)))
    np.testing.assert_allclose(tangent, exact_tangent, rtol=0.0, atol=_TANGENT * scale)


# measured 1.2e-8 (time) and 1.3e-7 (tangent) relative to the closed form, CPU VM 2026-10-22,
# pytest tests/integration/test_native_backends.py::test_a_klu_projected_event_time_has_the_exact_implicit_derivative
_EVENT = 1e-6


@pytest.mark.requires_klu
def test_a_klu_projected_event_time_has_the_exact_implicit_derivative():
    """An event on the algebraic global ``lam = e^{-2kt}``, located on projected states over
    KLU, fires at ``ln(4/3) / 2k`` and its tangent in ``k`` is ``-t / k``."""
    case, expected = ConstrainedDAE(), np.log(4.0 / 3.0) / (2.0 * float(_THETA))
    event = ds.Event(
        cond_fn=lambda t, state, args, **kwargs: state[-1] - 0.75,
        root_finder=_ROOT,
        direction="down",
        state_policy=_PROJECTED,
    )

    def event_time(theta):
        return ds.pdesolve(
            case.problem,
            ds.JIDA.stock(linear="klu", **_FINE),
            (0.0, 1.0),
            case.initial_state(),
            theta=theta,
            event=event,
            saveat=ds.SaveAt(t1=True),
            consistent_ic=True,
            max_steps=4000,
        ).terminal.t

    time, tangent = jax.jvp(event_time, (_THETA,), (jnp.ones_like(_THETA),))
    np.testing.assert_allclose([time, tangent], [expected, -expected / float(_THETA)], rtol=_EVENT)


# measured 3.1e-9 (value) and 1.8e-8 (gradient) relative to the closed form on KLU at rtol
# 1e-9, CPU VM 2026-10-22, pytest tests/integration/test_native_backends.py::test_held_factor_gradients_stay_exact_under_replay[klu-fp64];
# the cuDSS rows are owed (fp32_ir measured 5.3e-8 against the dense route on an L40S)
_HELD = {"fp64": 1e-7, "fp32_ir": 1e-6}


@pytest.mark.parametrize(
    ("linear", "mode"),
    [
        pytest.param("klu", "fp64", marks=pytest.mark.requires_klu),
        pytest.param("cudss", "fp64", marks=pytest.mark.requires_cudss),
        pytest.param("cudss", "fp32_ir", marks=pytest.mark.requires_cudss),
    ],
)
def test_held_factor_gradients_stay_exact_under_replay(linear, mode):
    """A checkpointed reverse pass replays segments out of order, each restored held factor
    rebuilt by its first solve: the loss and its gradient equal the closed form."""

    def loss(theta):
        solver = ds.JIDA.stock(
            linear=linear,
            precision=ds.Precision(solve_mode=mode),
            adjoint=ds.CheckpointAdjoint(checkpoints=3),
            **{"rtol": 1e-9, "atol": 1e-11},
        )
        solution = ds.pdesolve(
            _DECAY.problem,
            solver,
            (0.0, 0.5),
            _DECAY.initial_state(),
            theta=theta,
            saveat=ds.SaveAt(t1=True),
            max_steps=1000,
        )
        return jnp.sum(solution.ys[-1] ** 2)

    expected = jax.value_and_grad(lambda k: jnp.sum(_DECAY.analytical(0.5, k) ** 2))(_THETA)
    np.testing.assert_allclose(jax.value_and_grad(loss)(_THETA), expected, rtol=_HELD[mode])


_DIRECT = ds.MassProblem(
    rhs=lambda t, y, theta, control: -theta * y,
    mass=ds.MassOperator.identity(1),
    structure=ds.DAEStructure(differential_vars=np.ones((1,), dtype=bool)),
)
_STEPPER, _K, _Y0 = ds.DiffraxStepper(**{"rtol": 1e-8, "atol": 1e-8}), jnp.asarray(0.7), jnp.ones(1)
# measured 2.8e-10 relative at the end of the direct decay, 2.2e-16 on the pinned boundary
# and its unseeded offset, 1.7e-10 relative to the exact interior decay, CPU VM 2026-10-22,
# pytest tests/integration/test_native_backends.py::test_the_diffrax_stepper_solves_what_it_is_eligible_for_and_names_the_rest
_DIRECT_BOUND, _PINNED, _INTERIOR = 1e-8, 1e-12, 1e-8


def _pinned():
    spec = ds.problem("diffrax_pinned").domain("line", cartesian_grid_1d(3, 0.0, 1.0, method="fvm"))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: 0.25 + 0.25 * x)
    spec.equation(writes="u", reads=("u",))(lambda q: -q.u)
    spec.bc.dirichlet("u", node_indices=np.asarray([0]), value_fn=lambda time: 1.0 + 2.0 * time)
    return spec.build(), spec.assemble_state({})


def test_the_diffrax_stepper_solves_what_it_is_eligible_for_and_names_the_rest():
    """A differential problem solves to the exact decay with Diffrax's statistics; a pinned
    boundary integrates its differentiated value (from a seeded or an offset start) and the
    interior decays exactly; algebraic rows, a non-unit mass, native-only keywords,
    one-sided or affecting events, dense objects, projected output and counted steps are
    refused naming the compatible path."""
    solution = ds.pdesolve(
        _DIRECT,
        _STEPPER,
        (0.0, 0.4),
        _Y0,
        theta=_K,
        saveat=ds.SaveAt(t0=True, t1=True),
        dt0=0.01,
        max_steps=100,
    )
    np.testing.assert_array_equal(solution.ts, [0.0, 0.4])
    np.testing.assert_allclose(solution.ys[-1, 0], jnp.exp(-0.4 * _K), rtol=_DIRECT_BOUND)
    assert bool(solution.ok) and solution.event_log is None
    assert solution.info["diffrax"]["num_steps"] > 0
    pinned, start = _pinned()
    times = jnp.asarray([0.0, 0.1, 0.2])
    tracks = {}
    for seeded in (True, False):
        keywords = {} if seeded else {"consistent_ic": False}
        tracks[seeded] = ds.pdesolve(
            pinned, _STEPPER, (0.0, 0.2), start, saveat=ds.SaveAt(ts=times), **keywords
        ).ys
        offset = 0.0 if seeded else float(start[0]) - 1.0
        np.testing.assert_allclose(tracks[seeded][:, 0] - (1.0 + 2.0 * times), offset, atol=_PINNED)
    exact = start[1:][None, :] * jnp.exp(-times)[:, None]
    np.testing.assert_allclose(tracks[True][:, 1:], exact, rtol=_INTERIOR)
    mass = ds.MassOperator.sparse([0], [0], [1.0], n=2)
    algebraic = ds.DAEStructure(differential_vars=np.asarray([True, False]))
    refused = [
        (
            ds.MassProblem(
                rhs=lambda t, y, theta, control: 0.0 * y, mass=mass, structure=algebraic
            ),
            jnp.ones(2),
            {},
            "algebraic local",
        ),
        (
            ds.MassProblem(
                rhs=lambda t, y, theta, control: -y,
                mass=ds.MassOperator.diagonal(jnp.asarray([2.0])),
                structure=_DIRECT.structure,
            ),
            _Y0,
            {},
            "non-unit",
        ),
    ]
    for keywords, match in (
        ({"lambda0": jnp.zeros((1,))}, "lambda0"),
        ({"linear": "dense"}, "linear-backend"),
        ({"hmin": 1e-8}, "native-engine option"),
        ({"control_state0": {}}, "control_state0"),
        ({"saveat": ds.SaveAt(t1=True, dense=True)}, "dense object"),
        ({"saveat": ds.SaveAt(output=_PROJECTED)}, "projected output"),
        ({"saveat": ds.SaveAt(t1=True, steps=3)}, "boolean form"),
        (
            {"event": ds.Event(lambda t, y, args, **k: t - 0.2, root_finder=_ROOT, direction="up")},
            "up/down crossing",
        ),
        (
            {
                "event": ds.Event(
                    lambda t, y, args, **k: t > 0.1, affect=lambda t, y, args, control: (y, control)
                )
            },
            "event affect",
        ),
    ):
        refused.append((_DIRECT, _Y0, {"theta": _K, **keywords}, match))
    for problem, y0, keywords, match in refused:
        with pytest.raises(ds.ConfigurationError, match=match):
            ds.pdesolve(
                problem, _STEPPER, (0.0, 0.1), y0, **{"saveat": ds.SaveAt(t1=True)} | keywords
            )


# measured 6.1e-9 relative for both adjoints against the closed form, CPU VM 2026-10-22,
# pytest tests/integration/test_native_backends.py::test_the_diffrax_stepper_differentiates_under_the_forward_and_checkpoint_adjoints
_ADJOINT = 1e-7


def test_the_diffrax_stepper_differentiates_under_the_forward_and_checkpoint_adjoints():
    """Parameters travel through Diffrax's arguments: the forward and the checkpoint adjoint
    give ``d/dk e^{-0.4 k} = -0.4 e^{-0.4 k}``."""

    def endpoint(theta, adjoint=None):
        return ds.pdesolve(
            _DIRECT,
            _STEPPER,
            (0.0, 0.4),
            _Y0,
            theta=theta,
            adjoint=adjoint,
            saveat=ds.SaveAt(t1=True),
        ).ys[-1, 0]

    expected = -0.4 * jnp.exp(-0.4 * _K)
    forward = jax.jacfwd(lambda k: endpoint(k, ds.ForwardAdjoint()))(_K)
    np.testing.assert_allclose([forward, jax.grad(endpoint)(_K)], expected, rtol=_ADJOINT)


# measured 1.1e-12 from the event time 0.2 (1.3e-12 from JNDF's), CPU VM 2026-10-22,
# pytest tests/integration/test_native_backends.py::test_diffrax_events_and_step_outputs_read_like_the_native_engines
_EVENT_TIME = 1e-9


def test_diffrax_events_and_step_outputs_read_like_the_native_engines():
    """A terminating event stops Diffrax at its root with EVENT_STOP, a NaN-padded prefix and
    the terminal state in the reserved row, as the native engines do; saving every step records each
    accepted step once and ends at the terminal state."""
    event = ds.Event(lambda t, y, args, **k: t - 0.2, root_finder=_ROOT, direction="any")
    saveat = ds.SaveAt(ts=jnp.asarray([0.0, 0.1, 0.3]))
    stopped = ds.pdesolve(_DIRECT, _STEPPER, (0.0, 0.4), _Y0, theta=_K, event=event, saveat=saveat)
    assert stopped.result == ds.RESULT.EVENT_STOP
    np.testing.assert_allclose(stopped.terminal.t, 0.2, atol=_EVENT_TIME)
    np.testing.assert_array_equal(stopped.valid_mask, [True, True, True, False])
    assert bool(jnp.isnan(stopped.ts[-1])) and int(stopped.event_kinds[2]) == 2
    np.testing.assert_array_equal(stopped.ys[2], stopped.terminal.y)
    steps = ds.pdesolve(
        _DIRECT, _STEPPER, (0.0, 0.4), _Y0, theta=_K, saveat=ds.SaveAt(t0=True, steps=True)
    )
    count = int(steps.valid_count)
    assert count == int(steps.stats.accepts) + 1 and bool(jnp.all(jnp.diff(steps.ts[:count]) > 0))
    np.testing.assert_array_equal(steps.ys[count - 1], steps.terminal.y)


def _cahn_hilliard(dae):
    """``(problem, initial(theta), weights)`` of a periodic Cahn-Hilliard chain, with the
    chemical potential as an algebraic unknown when ``dae``."""
    n, rng = (64 if dae else 128), np.random.default_rng(3)
    index, c0 = np.arange(n), jnp.asarray(0.05 * rng.standard_normal(n))

    def laplacian(u):
        return (jnp.roll(u, 1) - 2.0 * u + jnp.roll(u, -1)) * n**2

    if not dae:
        offsets = (-2, -1, 0, 1, 2)
        rows, cols = np.tile(index, 5), np.concatenate([(index + o) % n for o in offsets])
        problem = ds.MassProblem(
            rhs=lambda t, c, th, control: laplacian(th[0] * (c**3 - c) - th[1] * laplacian(c)),
            mass=ds.MassOperator.identity(n),
            structure=ds.DAEStructure(differential_vars=np.ones((n,), dtype=bool)),
            rhs_sparsity=SparseTopology(rows=rows, cols=cols, n_dofs=n),
        )
        return problem, lambda th: c0, jnp.asarray(rng.standard_normal(n))
    near = [(index + o) % n for o in (-1, 0, 1)]
    rows = np.concatenate([index] * 3 + [n + index] * 4)
    cols = np.concatenate([n + k for k in near] + [n + index] + near)

    def rhs(t, y, th, control):
        c, mu = y[:n], y[n:]
        return jnp.concatenate([laplacian(mu), mu - th[0] * (c**3 - c) + th[1] * laplacian(c)])

    problem = ds.MassProblem(
        rhs=rhs,
        mass=ds.MassOperator.sparse(index, index, np.ones(n), n=2 * n),
        structure=ds.DAEStructure(differential_vars=np.arange(2 * n) < n),
        rhs_sparsity=SparseTopology(rows=rows, cols=cols, n_dofs=2 * n),
    )

    def initial(th):
        return jnp.concatenate([c0, th[0] * (c0**3 - c0) - th[1] * laplacian(c0)])

    return problem, initial, jnp.asarray(rng.standard_normal(2 * n))


@pytest.mark.requires_cudss
@pytest.mark.parametrize("dae", [False, True], ids=["differential", "dae"])
@pytest.mark.parametrize("engine", ["JNDF", "JBDF", "JIDA"])
def test_cudss_engines_factor_once_per_setup_and_replay_exactly(engine, dae):
    """Every corrector setup of a steady forward solve refactors the shared storage in place
    (or finds its matrix held), nothing is rebuilt and no slot is added; a checkpointed
    reverse pass rebuilds nothing, runs no new analysis, solves each attempted step's cotangent
    in transpose mode (on a differential problem by one factorization per step and no other)
    and equals the forward-mode derivative to rounding."""
    from diff_soup.linear import CuDSS

    problem, initial, weights = _cahn_hilliard(dae)
    theta = jnp.asarray([1.0, 2e-4])

    def solve(th, adjoint=None):
        solver = getattr(ds, engine).stock(
            linear="cudss", adjoint=adjoint, **{"rtol": 1e-6, "atol": 1e-8}
        )
        span = (0.0, 0.002 if dae else 0.01)
        return ds.pdesolve(problem, solver, span, initial(th), theta=th, max_steps=2000)

    def counted(program, *args):
        jax.block_until_ready(program(*args))
        before = CuDSS.contract_stats()
        out = jax.block_until_ready(program(*args))
        return out, {k: v - before[k] for k, v in CuDSS.contract_stats().items()}

    (_, setups, attempts), delta = counted(
        jax.jit(
            lambda th: (lambda s: (s.ys[-1], s.stats.factorizations, s.stats.trial_steps))(
                solve(th)
            )
        ),
        theta,
    )
    assert int(setups) > 2
    assert delta["refactorizations"] + delta["storage_asks_unchanged"] == int(setups)
    assert (delta["rebuilds"], delta["live_slots"]) == (0, 0)

    def loss(th, adjoint):
        return jnp.sum(weights * solve(th, adjoint).ys[-1])

    reverse, delta = counted(
        jax.jit(jax.grad(lambda th: loss(th, ds.CheckpointAdjoint(checkpoints=3)))), theta
    )
    tangent = jax.jit(jax.jacfwd(lambda th: loss(th, ds.ForwardAdjoint())))(theta)
    # measured: owed with this cuDSS row (no CPU value exists); 1e-8 bounds the replay
    # against the forward-mode derivative to rounding
    np.testing.assert_allclose(reverse, tangent, rtol=1e-8)
    assert delta["rebuilds"] == 0
    # measured per pullback on an RTX 4070 Ti, cuDSS 1.0: one factorization and one solve in
    # transpose mode per attempted step on the differential rows (JNDF 331, JBDF 424, JIDA
    # 462); the dae rows solve 132, 138 and 145 times in transpose mode for 130, 137 and 144
    # attempts and factor 144, 143 and 146 times, the derivative of the start included
    assert delta["transposed_solves"] >= int(attempts)
    if not dae:
        assert delta["factorizations"] == delta["transposed_solves"] == int(attempts)
    assert delta["cold_analyses"] + delta["seeded_analyses"] == 0
