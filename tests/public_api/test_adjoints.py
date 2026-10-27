"""Adjoints: reverse-mode gradients through an affect, forward and checkpoint parity, the
checkpoint record, second order through an event solve, and the compositions of jax.jvp,
jax.grad, jax.jit and jax.vmap over a checkpointed solve."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.timestepping import Bisection
from tests.support.problems import ConstrainedDAE, DecayODE

_THETA = jnp.asarray(0.4)
#: Solver and root-finder settings (requested tolerances, not asserted bounds).
_TIGHT = {"rtol": 1e-8, "atol": 1e-10}
_VERY_TIGHT = {"rtol": 1e-10, "atol": 1e-12}
_ROOT = {"rtol": 1e-10, "atol": 1e-12}
_ODE = DecayODE()
# measured relative gradient errors 3.9e-7 (affect, two checkpoints),
# CPU VM 2026-10-22, pytest tests/public_api/test_adjoints.py::test_a_two_checkpoint_gradient_through_an_affect_is_the_closed_form
_GRADIENT_BOUND = 5e-6
# measured 4.2e-16 relative between the two adjoints, CPU VM 2026-10-22,
# pytest tests/public_api/test_adjoints.py::test_a_forward_adjoint_tangent_equals_the_checkpoint_gradient
_PARITY_BOUND = 1e-12
# measured 3.3e-8 relative against the closed-form Hessian, CPU VM 2026-10-22,
# pytest tests/public_api/test_adjoints.py::test_a_hessian_through_an_event_solve_matches_the_closed_form
_HESSIAN_BOUND = 5e-7


def _halve_once_event():
    def condition(time, state, args, *, control=None, **kwargs):
        return jnp.where(control["armed"] > 0, time - 0.025, -1.0)

    def halve(time, state, args, control):
        return 0.5 * state, {"armed": jnp.asarray(0, jnp.int32)}

    return ds.Event(condition, halve, Bisection(**_ROOT), "up")


_HALVE = _halve_once_event()


def _affect_endpoint(theta, adjoint):
    solution = ds.pdesolve(
        _ODE.problem,
        ds.JNDF.stock(linear="dense", **_TIGHT),
        (0.0, 0.05),
        _ODE.initial_state(),
        theta=theta,
        adjoint=adjoint,
        control_state0={"armed": jnp.asarray(1, jnp.int32)},
        event=_HALVE,
        max_steps=300,
    )
    return solution.ys[-1, 0], solution.stats


@pytest.fixture(scope="module")
def affect():
    """The two-checkpoint reverse-mode derivative of a solve with an affect, and its stats."""
    adjoint = ds.CheckpointAdjoint(checkpoints=2)
    return jax.grad(lambda th: _affect_endpoint(th, adjoint), has_aux=True)(_THETA)


def test_a_two_checkpoint_gradient_through_an_affect_is_the_closed_form(affect):
    """Reverse mode with two online checkpoints differentiates a mutate-and-continue affect
    that halves the state: the gradient is ``-0.5 t e^{-k t}``."""
    np.testing.assert_allclose(affect[0], -0.025 * np.exp(-0.4 * 0.05), rtol=_GRADIENT_BOUND)


def test_a_checkpointed_solve_records_its_adjoint_and_checkpoint_count(affect):
    """The statistics of a reverse-mode solve name the adjoint and the checkpoints it ran."""
    stats = affect[1]
    assert (stats.adjoint, stats.adjoint_checkpoints) == ("CheckpointAdjoint", 2)


def test_a_forward_adjoint_tangent_equals_the_checkpoint_gradient(affect):
    """jax.jvp under an explicit ForwardAdjoint gives the derivative the CheckpointAdjoint
    gradient gives, through the same affect."""
    tangent = jax.jvp(
        lambda th: _affect_endpoint(th, ds.ForwardAdjoint())[0], (_THETA,), (jnp.ones_like(_THETA),)
    )[1]
    np.testing.assert_allclose(tangent, affect[0], rtol=_PARITY_BOUND)


_LEVEL = 0.93
_CROSSING = ds.Event(
    cond_fn=lambda time, state, args, **kwargs: state[0] - _LEVEL,
    root_finder=Bisection(**_ROOT),
    direction="down",
)


def test_a_hessian_through_an_event_solve_matches_the_closed_form():
    """jax.hessian under CheckpointAdjoint of ``|y(t*)|^2 + t*`` at the crossing
    ``y0 = 0.93`` is ``2 ln(1/0.93) / k^3``."""

    def loss(theta):
        solution = ds.pdesolve(
            _ODE.problem,
            ds.JNDF.stock(linear="dense", **_VERY_TIGHT),
            (0.0, 1.0),
            _ODE.initial_state(),
            theta=theta,
            adjoint=ds.CheckpointAdjoint(),
            event=_CROSSING,
            max_steps=2000,
        )
        return jnp.sum(solution.terminal.y**2) + solution.terminal.t

    expected = 2.0 * np.log(1.0 / _LEVEL) / 0.4**3
    np.testing.assert_allclose(jax.hessian(loss)(_THETA), expected, rtol=_HESSIAN_BOUND)


_MASS_Y0, _MASS_THETA = jnp.asarray([1.0, 0.5]), jnp.asarray([1.3, 0.7])
_DIRECTION = jnp.asarray([1.0, -0.5])
#: The central-difference step of the gradient along ``_DIRECTION``.
_STEP = 1e-6
# The Hessian-vector product against the central difference of the gradient, relative in the
# max norm (the difference also moves the adaptive step schedule the derivative holds fixed):
# measured 1.2e-7 (JNDF, dense; 2.5e-8 JIDA on KLU and 9.3e-7 JBDF, not rows for their compile
# time), CPU VM 2026-10-22,
# pytest tests/public_api/test_adjoints.py::test_a_forward_push_over_the_jitted_gradient_is_the_hessian_vector_product
_HVP_BOUND = 5e-6
# measured 2.0e-14 between the batched gradient and each element's own (JNDF, dense), CPU VM
# 2026-10-22, pytest tests/public_api/test_adjoints.py::test_a_vmapped_gradient_equals_each_elements_gradient
_COMPOSED_BOUND = 1e-12
_MASS_PROBLEM = ds.MassProblem(
    rhs=lambda t, y, theta, control: jnp.stack(
        [-theta[0] * y[0] * y[1] + jnp.sin(t), y[0] - y[1] ** 2 + theta[1]]
    ),
    mass=ds.MassOperator.diagonal(
        lambda t, y, theta, control: jnp.asarray([1.0, 2.0]),
        n=2,
        dependencies=ds.Dependencies(time=False, state=False, parameters=False, control=False),
    ),
    structure=ds.DAEStructure(differential_vars=np.array([True, True])),
)


def _squared_end(engine, linear="dense"):
    """The squared end state of a small stiff checkpointed solve, as a function of theta."""
    solver = engine.stock(linear=linear, **_VERY_TIGHT)

    def loss(theta):
        solution = ds.pdesolve(
            _MASS_PROBLEM,
            solver,
            (0.0, 0.3),
            _MASS_Y0,
            theta=theta,
            adjoint=ds.CheckpointAdjoint(),
            saveat=ds.SaveAt(t1=True),
            max_steps=2000,
        )
        return jnp.sum(solution.ys[-1] ** 2)

    return loss


def _relative(got, want):
    """The max-norm distance of ``got`` from ``want``, relative to ``want``."""
    got, want = np.asarray(got), np.asarray(want)
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


@functools.cache
def _jitted_gradient(engine, linear="dense"):
    """One jitted gradient per engine and backend, so the two compositions below share one
    compiled program."""
    return jax.jit(jax.grad(_squared_end(engine, linear)))


def test_a_forward_push_over_the_jitted_gradient_is_the_hessian_vector_product():
    """``jax.jvp(jax.jit(jax.grad(loss)))`` under CheckpointAdjoint equals the central
    difference of the gradient."""
    gradient = _jitted_gradient(ds.JNDF)
    _, hvp = jax.jvp(gradient, (_MASS_THETA,), (_DIRECTION,))
    ahead = np.asarray(gradient(_MASS_THETA + _STEP * _DIRECTION))
    behind = np.asarray(gradient(_MASS_THETA - _STEP * _DIRECTION))
    assert _relative(hvp, (ahead - behind) / (2.0 * _STEP)) <= _HVP_BOUND


def test_a_vmapped_gradient_equals_each_elements_gradient():
    """``jax.vmap(jax.grad(loss))`` over a parameter batch equals the per-element gradients."""
    batch = jnp.stack([_MASS_THETA, 1.1 * _MASS_THETA, 0.9 * _MASS_THETA])
    gradient = _jitted_gradient(ds.JNDF)
    each = np.stack([np.asarray(gradient(theta)) for theta in batch])
    assert _relative(jax.vmap(gradient)(batch), each) <= _COMPOSED_BOUND


def _global_decay():
    """Two decaying cells beside a differential global ``z' = -rate z`` from ``z = 1``."""
    spec = ds.problem("differential_global_decay").domain(
        "line", ds.cartesian_grid_1d(2, 0.0, 1.0, method="fvm")
    )
    rate = spec.param("rate", value=0.7, learnable=True)
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
    spec.equation(writes="u", reads=("u",))(lambda q: -q.u)
    spec.global_var("z", initial=1.0, mass="differential")
    spec.equation(writes="z", reads=("z",))(lambda q: -q.p.rate * q["z"])
    return spec.build(), spec.assemble_state({}), rate


# measured 1.9e-9 relative against the closed form, workstation 2026-10-23,
# pytest tests/public_api/test_adjoints.py::test_the_gradient_of_a_differential_global_through_a_solve_is_the_closed_form
_GLOBAL_BOUND = 1e-7


def test_the_gradient_of_a_differential_global_through_a_solve_is_the_closed_form():
    """Reverse mode under CheckpointAdjoint differentiates a recipe's differential global
    ``z' = -rate z``: the gradient of ``z(1)`` in ``rate`` is ``-e^{-rate}``."""
    problem, start, rate = _global_decay()

    def global_at_one(value):
        return ds.pdesolve(
            problem,
            ds.JNDF.stock(linear="dense", **_VERY_TIGHT),
            (0.0, 1.0),
            start,
            theta=ds.default_theta(problem, {rate: value}),
            adjoint=ds.CheckpointAdjoint(),
            saveat=ds.SaveAt(t1=True),
            max_steps=2000,
        ).lambdas[-1, 0]

    gradient = jax.grad(global_at_one)(jnp.asarray(0.7))
    np.testing.assert_allclose(gradient, -np.exp(-0.7), rtol=_GLOBAL_BOUND)


#: A terminating event on the algebraic global ``lam = e^{-2kt}`` of the index-one DAE,
#: located on projected states; it fires at ``t* = ln(4/3) / 2k``.
_PROJECTED_STOP = ds.Event(
    cond_fn=lambda time, state, args, **kwargs: state[-1] - 0.75,
    root_finder=Bisection(**_ROOT),
    direction="down",
    state_policy=ds.DenseOutputPolicy(mode="projected"),
)
# measured 2.7e-9 relative against the closed form, workstation 2026-10-23,
# pytest tests/public_api/test_adjoints.py::test_the_gradient_of_a_projected_dae_event_time_is_the_closed_form
_EVENT_GRADIENT_BOUND = 1e-7


def test_the_gradient_of_a_projected_dae_event_time_is_the_closed_form():
    """Reverse mode under CheckpointAdjoint through a terminating event located on projected
    DAE states gives ``dt*/dk = -t*/k``."""
    case = ConstrainedDAE()

    def event_time(theta):
        return ds.pdesolve(
            case.problem,
            ds.JNDF.stock(linear="dense", **_TIGHT),
            (0.0, 1.0),
            case.initial_state(),
            theta=theta,
            adjoint=ds.CheckpointAdjoint(),
            event=_PROJECTED_STOP,
            saveat=ds.SaveAt(t1=True),
            max_steps=2000,
        ).terminal.t

    crossing = np.log(4.0 / 3.0) / (2.0 * float(_THETA))
    gradient = jax.grad(event_time)(_THETA)
    np.testing.assert_allclose(gradient, -crossing / float(_THETA), rtol=_EVENT_GRADIENT_BOUND)
