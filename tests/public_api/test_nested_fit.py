"""The nested least-squares fit: shared values over a batch of inner fits, posed once as a
least-squares problem over the shared values, against the joint fit and central differences."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import diff_soup as ds
from diff_soup.optimization import (
    InnerSteps,
    InteriorPoint,
    LeastSquaresProblem,
    NestedLeastSquares,
    NewtonTrustRegion,
    optsolve,
)

T = jnp.linspace(0.0, 2.0, 25)
W_TRUE = jnp.array([0.8, 2.0])
# The third member's second own value wants to be negative: its lower bound at 0 is active.
Z_TRUE = jnp.array([[1.0, 0.6], [1.5, 0.3], [0.7, -0.4]])
LOWER = jnp.array([-jnp.inf, 0.0])
X0 = jnp.ones((3, 2))
W0 = jnp.array([1.0, 1.7])
# measured: every member ends KKT_TOL, workstation CPU 2026-11-08
INNER = InteriorPoint(hessian="gauss-newton", tol=1e-12)

# measured 2.8e-9, workstation CPU 2026-11-08, pytest tests/public_api/test_nested_fit.py
_JOINT = 1e-8
# measured 9.0e-10 (forward and reverse), workstation CPU 2026-11-08, pytest tests/public_api/test_nested_fit.py
_GRADIENT = 1e-6


def _model(z, w):
    """Two own values and two shared ones, nonlinear in both."""
    return z[0] * jnp.exp(-w[0] * T) + z[1] * jnp.sin(w[1] * T)


def _data():
    """Each member's rows at the true values, with a little structured noise."""
    noise = 0.02 * jnp.sin(7.0 * T)[None, :] * jnp.array([1.0, -1.0, 0.5])[:, None]
    return jax.vmap(_model, in_axes=(0, None))(Z_TRUE, W_TRUE) + noise


def _nested(adjoint=None):
    """The posed nested fit."""
    data = _data()
    inner = LeastSquaresProblem.from_residual(
        jnp.zeros(2),
        lambda z, a: _model(z, a["shared"]) - a["y"],
        args={"shared": W0, "y": data[0]},
        bounds=(LOWER, None),
        hessian=False,
    )
    return NestedLeastSquares.pose(
        inner, INNER, {"y": data}, X0, W0, steps=InnerSteps(tol_min=1e-12), adjoint=adjoint
    )


def _joint_shared_values():
    """The joint fit over every member's own values and the shared ones, one interior point solve."""
    data = _data()

    def rows(v):
        z, w = v[:6].reshape(3, 2), v[6:]
        return (jax.vmap(_model, in_axes=(0, None))(z, w) - data).ravel()

    lower = jnp.concatenate([jnp.tile(LOWER, 3), jnp.full(2, -jnp.inf)])
    joint = LeastSquaresProblem.from_residual(
        jnp.concatenate([X0.ravel(), W0]), rows, bounds=(lower, None), hessian=False
    )
    # measured: ends KKT_TOL at 2.1e-14, workstation CPU 2026-11-08
    x, _ = InteriorPoint(hessian="gauss-newton", tol=1e-13, max_steps=500).run(
        joint, joint.x0, None
    )
    return x[6:]


def test_a_nested_fit_finds_the_joint_fits_shared_values():
    """`optsolve` of the nested problem with a Gauss-Newton trust region, the members' state
    carried from step to step, ends at the shared values of the joint fit."""
    nested = _nested()
    # The cost and step tests are off: the fit runs its 20 steps, past the joint fit's accuracy.
    outer = NewtonTrustRegion(
        hessian="gauss-newton", max_steps=20, grad_tol=1e-14, cost_tol=0.0, step_tol=0.0
    )
    fit = optsolve(nested.problem, outer, diagnostics=False)
    _, record = nested.solve(fit.params, fit.info["args"])
    assert int(record.status[2]) == ds.optimization.TerminationStatus.KKT_TOL
    assert float(jnp.max(jnp.abs(jnp.asarray(fit.params) - _joint_shared_values()))) <= _JOINT


@pytest.mark.parametrize("adjoint", ["forward", "reverse"])
def test_the_outer_gradient_matches_central_differences(adjoint):
    """`jax.grad` of the outer cost, by the forward rule transposed or by the reverse rule (one
    transposed solve per member), equals central differences of the cost."""
    rule = {"forward": ds.ForwardAdjoint(), "reverse": ds.nonlinear.ImplicitAdjoint()}[adjoint]
    value = _nested(rule).problem.value
    w, h = jnp.array([0.9, 1.9]), 1e-5
    central = jnp.stack([(value(w + h * e) - value(w - h * e)) / (2.0 * h) for e in jnp.eye(2)])
    gradient = jax.grad(value)(w)
    assert float(jnp.max(jnp.abs(gradient - central)) / jnp.max(jnp.abs(central))) <= _GRADIENT
