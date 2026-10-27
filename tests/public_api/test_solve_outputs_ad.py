"""Solve outputs under transformations: named outputs under jax.jit and jax.vmap, the
reverse-mode gradient through a terminal sample that replaced a requested row, and the Diffrax
backend's outputs mapping."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.domains import cartesian_grid_1d
from diff_soup.timestepping import Bisection
from tests.support.problems import observed_recipe

_OBSERVED, _OBSERVED_Y0, _RATE = observed_recipe()
#: Eight states and one global on 24 save times; the named outputs are two scalars.
_SAVE_TIMES = jnp.linspace(0.05, 1.0, 24)
_NAMED = ("mean_u", "load_gap")
_VALUE = jnp.asarray(1.3)
#: Solver and root-finder settings (requested tolerances, not asserted bounds).
_CHECKPOINTED = ds.JNDF.stock(
    linear="dense", adjoint=ds.CheckpointAdjoint(), **{"rtol": 1e-9, "atol": 1e-11}
)
_EVENT_ROOT = {"atol": 1e-12}
_DIFFRAX = {"rtol": 1e-9, "atol": 1e-9}
# measured 1.5e-13 (mean_u) and 3.5e-13 (load_gap) between the middle lane of the batch and the
# unbatched solve, CPU VM 2026-10-22,
# pytest tests/public_api/test_solve_outputs_ad.py::test_named_outputs_batch_over_a_parameter
_LANE_BOUND = 1e-11
#: The central-difference step in the rate.
_STEP = 1e-3
# measured 1.8e-6 relative between the gradient and the central difference at step 1e-3
# (2.2e-4 at 1e-5, where the adaptive schedule's noise dominates), CPU VM 2026-10-22,
# pytest tests/public_api/test_solve_outputs_ad.py::test_the_gradient_through_a_terminal_sample_that_replaced_a_row_is_the_difference
_GRADIENT_BOUND = 2e-5
# measured 2.2e-16 between the terminal states of the Diffrax solves, CPU VM 2026-10-22,
# pytest tests/public_api/test_solve_outputs_ad.py::test_the_diffrax_backend_stores_the_named_outputs_of_its_plain_save
_TERMINAL_BOUND = 1e-14


def _observed_solve(value, outputs, *, ts=_SAVE_TIMES, event=None):
    theta = ds.default_theta(_OBSERVED, {_RATE: value})
    return ds.pdesolve(
        _OBSERVED,
        _CHECKPOINTED,
        (0.0, 1.0),
        _OBSERVED_Y0,
        theta=theta,
        saveat=ds.SaveAt(ts=ts, outputs=outputs),
        event=event,
    )


@pytest.fixture(scope="module")
def named():
    """The named outputs of one unbatched, untransformed solve."""
    return {
        key: np.asarray(value) for key, value in _observed_solve(_VALUE, _NAMED).outputs.items()
    }


def test_named_outputs_are_unchanged_under_jit(named):
    """A jitted solve stores the named outputs the eager solve stores."""
    jitted = jax.jit(lambda value: dict(_observed_solve(value, _NAMED).outputs))(_VALUE)
    for name in _NAMED:
        np.testing.assert_array_equal(np.asarray(jitted[name]), named[name])


def test_named_outputs_batch_over_a_parameter(named):
    """``jax.vmap`` over the rate gives one lane of named outputs per value, and the lane at the
    unbatched value equals the unbatched solve."""
    values = jnp.asarray([0.9, 1.3, 1.7])
    batched = jax.vmap(lambda value: dict(_observed_solve(value, _NAMED).outputs))(values)
    assert all(batched[name].shape == (3, 24) for name in _NAMED)
    for name in _NAMED:
        np.testing.assert_allclose(batched[name][1], named[name], rtol=0.0, atol=_LANE_BOUND)


#: A terminating event at a requested save time: its sample replaces that row.
_STOP = ds.Event(
    cond_fn=lambda time, *args, **kwargs: time - 0.3, root_finder=Bisection(**_EVENT_ROOT)
)
_SHORT_TIMES = jnp.asarray([0.1, 0.2, 0.3, 0.5, 0.8])


def _collided(value):
    """The sum of the stored samples of a solve whose terminal sample replaced a row."""
    solution = _observed_solve(value, ("mean_u",), ts=_SHORT_TIMES, event=_STOP)
    return jnp.nansum(solution.outputs["mean_u"])


def test_the_gradient_through_a_terminal_sample_that_replaced_a_row_is_the_difference():
    """Reverse mode through a named output whose terminal sample replaced a requested row
    differentiates the stored value once: it equals the central difference."""
    gradient = float(jax.grad(_collided)(_VALUE))
    loss = jax.jit(_collided)
    difference = float((loss(_VALUE + _STEP) - loss(_VALUE - _STEP)) / (2.0 * _STEP))
    assert abs(gradient - difference) <= _GRADIENT_BOUND * abs(difference)


def _decay():
    """Three decaying cells under the identity mass, with one declared observation."""
    spec = ds.problem("observed_decay").domain("line", cartesian_grid_1d(3, 0.0, 1.0, method="fvm"))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: 1.0 + x)
    spec.equation(writes="u", reads=("u",))(lambda q: -0.7 * q.u)
    spec.observe.scalar(name="mean_u", fn=lambda q: jnp.mean(q.u))
    return spec.build(), spec.assemble_state({})


def test_the_diffrax_backend_stores_the_named_outputs_of_its_plain_save():
    """``DiffraxStepper`` stores a named observation equal to observe() on its plain save, and
    naming it leaves the terminal state unchanged."""
    decay, start = _decay()
    stepper = ds.DiffraxStepper(**_DIFFRAX)
    grid = jnp.asarray([0.1, 0.2, 0.4])
    stored = ds.pdesolve(
        decay, stepper, (0.0, 0.4), start, saveat=ds.SaveAt(ts=grid, outputs=("mean_u",))
    )
    plain = ds.pdesolve(decay, stepper, (0.0, 0.4), start, saveat=ds.SaveAt(ts=grid))
    np.testing.assert_array_equal(stored.outputs["mean_u"], jnp.mean(plain.ys, axis=1))
    np.testing.assert_allclose(stored.terminal.y, plain.terminal.y, rtol=0.0, atol=_TERMINAL_BOUND)
