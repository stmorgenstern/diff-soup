"""Fits: optsolve and constrained_optsolve on single, joint and lowered-study problems."""

from __future__ import annotations

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import diff_soup as ds
from diff_soup import inference as inf
from diff_soup.domains import cartesian_grid_1d
from tests.support.inference import (
    TRUE_RATE,
    decay_experiment,
    decay_recipe,
    decay_study,
    small_fit,
    solver,
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
    """LevenbergMarquardt on one experiment ends converged at the true rate, and on fitted window
    starts a Jacobian formed 8 directions at a time steps as the whole one does."""
    decay, _ = single
    assert lm_fit.converged
    np.testing.assert_allclose(float(lm_fit[decay.rate]), TRUE_RATE, atol=_RATE_ATOL)
    wide = decay_recipe("fits_decay_windows", nodes=12)
    windows = inf.InferenceProblem(
        decay_experiment(wide),
        parameters=wide.rate,
        trajectory=inf.Multishoot(inf.Windows.uniform(2)),
    )
    whole, tiled = (
        small_fit(windows, inf.LevenbergMarquardt(max_steps=20, jacobian_tile=tile))
        for tile in ("full", 8)
    )
    assert (whole.info["jacobian_tile"], tiled.info["jacobian_tile"]) == ("full", 8)
    # 13 decision variables, 11 steps each; measured 0.0 worst difference, workstation CPU
    # 2026-10-28, pytest tests/public_api/test_fits.py
    np.testing.assert_allclose(tiled.params, whole.params, rtol=0.0, atol=1e-12)


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


def _network_problem(name, *, blind_rate=False, n_experiments=1):
    """A reaction ``u' = f(u)`` fitted by a tanh network with two identical hidden units.

    Identical units have identical Jacobian columns, so the weights are not
    identifiable one by one. ``blind_rate`` adds a scalar ``rate`` the data cannot see.
    More than one experiment makes a joint fit, each from its own start.
    """
    network = eqx.nn.MLP("scalar", "scalar", 2, 1, activation=jnp.tanh, key=jax.random.PRNGKey(0))
    network = eqx.tree_at(
        lambda net: (net.layers[0].weight, net.layers[0].bias, net.layers[1].weight),
        network,
        (jnp.full((2, 1), 0.5), jnp.full((2,), 0.1), jnp.full((1, 2), -0.3)),
    )
    spec = ds.problem(name)
    spec.domain("line", cartesian_grid_1d(3, 0.0, 1.0, method="fvm"))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: 1.0 + x)
    handles = (spec.function("reaction", network, learnable=True, pointwise=True),)
    if blind_rate:
        handles += (spec.param("rate", value=0.3, learnable=True),)

    @spec.equation(writes="u", reads=("u",))
    def reaction(q):
        blind = 0.0 * q.p.rate if blind_rate else 0.0
        return q.fn.reaction(q.u) + blind

    problem = spec.build()
    times = jnp.linspace(0.0, 0.8, 9)

    def experiment(k):
        y0 = spec.assemble_state({}) + 0.2 * k
        data = jax.vmap(lambda t: (y0 - 0.4) * jnp.exp(-0.5 * t) + 0.4)(times)
        return inf.Experiment(
            problem, solver(), (0.0, 0.8), y0, (inf.Measurement("u", times, data),)
        )

    experiments = tuple(experiment(k) for k in range(n_experiments))
    return inf.InferenceProblem(
        experiments[0] if n_experiments == 1 else experiments, parameters=handles
    )


# Levenberg-Marquardt records the Jacobian it ends on; Optax records none, so
# the diagnostics build their own over the columns that are not network weights.
_NETWORK_FITS = pytest.mark.parametrize(
    ("n_experiments", "optimizer"),
    [(1, inf.LevenbergMarquardt(max_steps=3)), (2, inf.Optax(optax.adam(1e-2), max_steps=3))],
    ids=["one-experiment-lm", "two-experiments-optax"],
)


@_NETWORK_FITS
def test_a_network_fit_records_a_note_instead_of_an_identifiability_warning(
    n_experiments, optimizer
):
    """optsolve on network weights alone, one experiment or a joint, warns nothing and notes the skipped check."""
    problem = _network_problem(f"fits_network_{n_experiments}", n_experiments=n_experiments)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = inf.optsolve(problem, optimizer)
    assert any("learnable-function weight" in note for note in fit.diagnostics.warnings)
    # No column is tested, so no Jacobian is built or reported.
    assert fit.diagnostics.theta_singular_values is None


@_NETWORK_FITS
def test_a_blind_scalar_beside_a_network_still_raises_the_identifiability_warning(
    n_experiments, optimizer
):
    """A scalar parameter the data cannot see is named in the warning, network weights or not."""
    problem = _network_problem(
        f"fits_network_blind_{n_experiments}", blind_rate=True, n_experiments=n_experiments
    )
    with pytest.warns(inf.fitting.IdentifiabilityWarning, match="rate"):
        fit = inf.optsolve(problem, optimizer)
    # The report covers the one scalar column, not the seven network weights.
    assert fit.diagnostics.theta_singular_values.shape == (1,)
