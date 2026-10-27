"""Second-order AD through solves: the composable forward rule, Hessians under the checkpoint
adjoint, the replay guard, run-time failures on an asynchronous device, and a declared
reverse-sweep dtype under the checkpoint gradient and Hessian."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.timestepping.vsvo.common import driver
from diff_soup.timestepping.vsvo.engines import ida_engine
from tests.support.second_order import (
    PROBLEM,
    SPAN,
    THETA,
    Y0,
    composable_reference,
    gradient_and_hessian,
    loss,
)

# measured 9.4e-16 (dense) and 1.5e-15 (KLU) elementwise relative between the two routes and
# 1.2e-16 asymmetry (the owed cuDSS row is unmeasured), CPU VM 2026-10-22,
# pytest tests/contracts/test_second_order_ad.py::test_the_checkpoint_hessian_equals_the_composable_reverse_over_forward_hessian
_HESSIAN = 1e-8


@pytest.fixture(scope="module")
def composable():
    """The reverse-over-forward Hessian and the forward gradient of the solve under
    ``ForwardAdjoint(composable=True)``, the full-precision reference."""
    return composable_reference()


@pytest.mark.parametrize(
    "linear",
    [
        "dense",
        pytest.param("klu", marks=pytest.mark.requires_klu),
        pytest.param("cudss", marks=pytest.mark.requires_cudss),
    ],
)
def test_the_checkpoint_hessian_equals_the_composable_reverse_over_forward_hessian(
    composable, linear
):
    """jax.hessian under CheckpointAdjoint (forward entries pushed over the reverse sweep) and
    reverse over the composable forward rule are two independent second-order routes: their
    Hessians agree and are symmetric."""
    reference = composable[0]
    hessian = jax.hessian(loss(ds.CheckpointAdjoint(), linear))(THETA)
    np.testing.assert_allclose(reference, reference.T, rtol=_HESSIAN)
    np.testing.assert_allclose(hessian, reference, rtol=_HESSIAN)


def _drifting_engine():
    """A JIDA engine that reports a perturbed next step on a replay, as an incomplete decision
    record would."""
    base = ida_engine(ds.JIDA().profile, linear="dense", **{"rtol": 1e-8, "atol": 1e-10})

    class Drifting(type(base)):
        def attempt_step(self, carry, decision=None):
            new_carry, report = super().attempt_step(carry, decision)
            if decision is not None:
                report = eqx.tree_at(
                    lambda r: r.h_new,
                    report,
                    jnp.where(decision[0], 1.01 * report.h_new, report.h_new),
                )
            return new_carry, report

    return Drifting(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)})


def _departing(theta):
    config = driver.DriverConfig(
        adjoint=ds.CheckpointAdjoint(checkpoints=2), theta=theta, **{"rtol": 1e-8}
    )
    solution = driver.run_engine(_drifting_engine(), PROBLEM, *SPAN, Y0, ds.SaveAt(t1=True), config)
    return jnp.sum(solution.ys[-1] ** 2)


def test_a_replay_that_departs_from_the_forward_sweep_raises():
    """A replayed step that does not reach the time and step the forward sweep recorded makes
    the reverse pass raise instead of returning the gradient of another trajectory."""
    with pytest.raises(Exception, match="departed from the forward trajectory"):
        jax.block_until_ready(jax.grad(_departing)(THETA))


@pytest.mark.requires_cuda
def test_a_solve_that_fails_at_run_time_leaves_later_solves_working(composable):
    """On an asynchronous device a reverse pass that fails while it runs fails nothing after
    it: the Hessian computed after it equals the composable reference."""
    with pytest.raises(Exception, match="departed from the forward trajectory"):
        jax.block_until_ready(jax.grad(_departing)(THETA))
    # No jax.effects_barrier here: JAX keeps the runtime token of a failed program with host
    # callbacks, and its barrier raises that program's error again until a newly compiled one
    # replaces the token, whatever library the program came from.
    hessian = jax.hessian(loss(ds.CheckpointAdjoint()))(THETA)
    np.testing.assert_allclose(hessian, composable[0], rtol=_HESSIAN)


#: A precision declaring its own reverse-sweep dtype, beside the default one whose Hessian the
#: test above pins.
_ADJOINT_DTYPE = ds.Precision(adjoint_compute_dtype=np.dtype("float64"))
# Relative to the composable full-precision reference: measured gradient 1.1e-15 and Hessian
# 9.4e-16, CPU VM 2026-10-22,
# pytest tests/contracts/test_second_order_ad.py::test_a_reverse_sweep_dtype_composes_under_the_hessian
_POLICY_BOUND = 1e-12


def test_a_reverse_sweep_dtype_composes_under_the_hessian(composable):
    """Under CheckpointAdjoint a precision declaring its own reverse-sweep dtype gives the
    full-precision gradient, and the forward pushes over it (the columns of jax.hessian) equal
    the reference Hessian."""
    reference_hessian, reference_gradient = composable
    gradient = jax.grad(loss(ds.CheckpointAdjoint(), precision=_ADJOINT_DTYPE))
    value, hessian = gradient_and_hessian(gradient)
    np.testing.assert_allclose(value, reference_gradient, rtol=_POLICY_BOUND)
    np.testing.assert_allclose(hessian, reference_hessian, rtol=_POLICY_BOUND)
