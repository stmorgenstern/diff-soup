"""Fits: optsolve and constrained_optsolve on single, joint and lowered-study problems."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import diff_soup as ds
from diff_soup import inference as inf
from tests.support.inference import (
    TRUE_RATE,
    decay_experiment,
    decay_recipe,
    decay_study,
    small_fit,
)

# measured 2.8e-6 (LM, NTR, BFGS) off the true rate 0.6 at solver rtol 1e-6,
# CPU VM 2026-10-22, pytest tests/public_api/test_fits.py
_RATE_ATOL = 3e-5


@pytest.fixture(scope="module")
def single():
    decay = decay_recipe("fits_decay")
    return decay, inf.InferenceProblem(decay_experiment(decay), parameters=decay.rate)


@pytest.fixture(scope="module")
def lm_fit(single):
    _, problem = single
    return small_fit(problem, inf.LevenbergMarquardt(max_steps=20))


def test_levenberg_marquardt_recovers_the_rate_of_a_single_experiment(single, lm_fit):
    """LevenbergMarquardt on one experiment ends converged at the true rate."""
    decay, _ = single
    assert lm_fit.converged
    np.testing.assert_allclose(float(lm_fit[decay.rate]), TRUE_RATE, atol=_RATE_ATOL)


def test_the_result_reports_point_status_steps_handle_values_and_final_rows(single, lm_fit):
    """An OptResult carries its point, status, steps, handle lookup and the final rows."""
    decay, problem = single
    rows, jacobian = problem.data_residual_jacobian(jnp.asarray(lm_fit.params))
    np.testing.assert_array_equal(np.asarray(lm_fit.info["final_residual"]), np.asarray(rows))
    np.testing.assert_array_equal(np.asarray(lm_fit.info["final_jacobian"]), np.asarray(jacobian))
    assert lm_fit.params.shape == (1,) and lm_fit.initial_params[0] == problem.x0[0]
    assert isinstance(lm_fit.status, inf.TerminationStatus)
    assert 0 < lm_fit.n_steps <= lm_fit.max_steps == 20
    assert float(lm_fit[decay.rate]) == float(lm_fit.params[0])
    assert float(lm_fit.experiments["decay"][decay.rate]) == float(lm_fit.params[0])


@pytest.mark.parametrize(
    "optimizer",
    [
        inf.NewtonTrustRegion(hessian="gauss-newton", max_steps=20),
        inf.BFGS(max_steps=30, ad="forward"),
    ],
    ids=("newton-gauss-newton", "bfgs"),
)
def test_newton_and_quasi_newton_optimizers_reach_the_same_rate(single, optimizer):
    """NewtonTrustRegion and BFGS converge to the true rate on one experiment."""
    decay, problem = single
    fit = small_fit(problem, optimizer)
    assert fit.converged
    np.testing.assert_allclose(float(fit[decay.rate]), TRUE_RATE, atol=_RATE_ATOL)


def test_gradient_descent_lowers_the_cost_at_every_step(single):
    """GradientDescent with a small rate takes only descending steps."""
    _, problem = single
    fit = small_fit(problem, inf.GradientDescent(lr=2e-5, max_steps=20, ad="forward"))
    assert fit.status is inf.TerminationStatus.MAX_STEPS
    assert np.all(np.diff(np.asarray(fit.cost_history)) < 0.0)


def test_optax_reports_cost_and_gradient_at_the_returned_parameters(single):
    """The Optax adapter's cost and projected gradient norm belong to its returned point."""
    _, problem = single
    fit = small_fit(problem, inf.Optax(optax.adam(0.05), max_steps=5, grad_tol=0.0, ad="forward"))
    point = jnp.asarray(fit.params)
    cost, gradient = problem.loss_and_grad_fn("forward")(point)
    projected = np.abs(np.asarray(point - jnp.clip(point - gradient, 0.1, 1.2)))
    # measured 0.0 relative difference for both, CPU VM 2026-10-22, pytest tests/public_api/test_fits.py
    np.testing.assert_allclose(fit.cost, float(cost), rtol=1e-12)
    np.testing.assert_allclose(fit.grad_norm, float(np.linalg.norm(projected)), rtol=1e-12)


def test_constrained_optsolve_meets_an_active_constraint_on_weighted_rows(single):
    """The default penalty reaches the constrained optimum of rows weighted by 1/sigma = 50."""
    _, problem = single
    result = inf.constrained_optsolve(
        problem,
        inf.LevenbergMarquardt(max_steps=40),
        constraints=inf.InequalityConstraints(
            g=lambda theta: jnp.atleast_1d(jax.tree_util.tree_leaves(theta)[0] - 0.5),
            n_constraints=1,
        ),
    )
    assert result.converged
    # measured 4.5e-7 off the active bound, CPU VM 2026-10-22, pytest tests/public_api/test_fits.py
    np.testing.assert_allclose(result.result.params, [0.5], atol=5e-6)


@pytest.fixture(scope="module")
def joint():
    decay = decay_recipe("fits_joint", source=True)
    return decay, decay_study(decay, noisy=False)


def test_a_joint_fit_keeps_every_coordinate_inside_its_bounds(joint):
    """A joint fit whose data pull one owner past its bound ends on that bound."""
    decay, _ = joint
    pulled = decay_study(decay, noisy=False)
    experiments = (*pulled.experiments[:2], decay_experiment(decay, "cell-c", source=0.4))
    study = inf.Study(
        experiments, parameters=pulled.parameters, clusters=pulled.clusters, independent=True
    )
    fit = small_fit(inf.InferenceProblem(study), inf.LevenbergMarquardt(max_steps=30))
    sources = fit[decay.source]
    assert float(sources["cell-c"]) == 0.25
    assert all(-0.25 <= float(value) <= 0.25 for value in sources.values())


def test_trust_region_on_a_lowered_study_recovers_shared_and_owner_parameters(joint):
    """ds.lower(study) fitted with TrustRegion recovers the rate and every owner's source."""
    decay, study = joint
    fit = small_fit(ds.lower(study), inf.TrustRegion(max_steps=20))
    assert fit.converged
    # measured 9.3e-6 on the rate, 7.9e-6 on a source, CPU VM 2026-10-22,
    # pytest tests/public_api/test_fits.py
    np.testing.assert_allclose(float(fit[decay.rate]), TRUE_RATE, atol=1e-4)
    sources = [float(fit[decay.source][owner]) for owner in ("cell-a", "cell-b", "cell-c")]
    # measured 7.9e-6, the run above
    np.testing.assert_allclose(sources, [-0.05, 0.0, 0.05], atol=1e-4)
