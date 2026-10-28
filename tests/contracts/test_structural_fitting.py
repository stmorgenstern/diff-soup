"""Structural fitting: optimizers on residual problems, KKT points, forward Jacobians, and the
trust-region loops' stop rules relative to the cost scale."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup import inference as inf
from diff_soup.inference.approx import KLBasis
from diff_soup.inference.solvers._loop import (
    Trials,
    _after_attempt,
    _floor_sample,
    _floor_sigma,
    _judge,
    cost_scale,
    cost_threshold,
)
from diff_soup.inference.solvers.tangent import (
    jacfwd_from_pushforward,
    linearize_value,
    value_and_jacfwd,
    value_and_jacfwd_tiled,
)
from tests.support.inference import solver
from tests.support.programs import primitive_names

_RNG = np.random.default_rng(0)
_MATRIX = _RNG.standard_normal((8, 3))
_TARGET = _RNG.standard_normal(8)
_EXACT = np.linalg.lstsq(_MATRIX, _TARGET, rcond=None)[0]

# measured 2.1e-8 worst parameter error (LevenbergMarquardt; BFGS 1.9e-10, Newton 3e-16),
# CPU VM 2026-10-22,
# pytest tests/contracts/test_structural_fitting.py
_LSQ_ATOL = 2e-7


@pytest.mark.parametrize(
    "optimizer",
    [
        inf.LevenbergMarquardt(max_steps=50, grad_tol=1e-10, linear_solver="qr"),
        inf.BFGS(max_steps=100, grad_tol=1e-10),
        inf.NewtonTrustRegion(hessian="true", subproblem="exact", max_steps=50),
        inf.NewtonTrustRegion(
            hessian="gauss-newton", subproblem="cg", x_scale=inf.ConstantScale(1.0), max_steps=50
        ),
    ],
    ids=("lm-qr", "bfgs", "newton-exact", "gauss-newton-cg"),
)
def test_optsolve_recovers_the_analytical_least_squares_solution(optimizer):
    """Every optimizer ends at the normal-equation solution of a linear residual."""
    problem = inf.LeastSquaresProblem.from_residual(
        np.zeros(3), lambda x: jnp.asarray(_MATRIX) @ x - jnp.asarray(_TARGET)
    )
    fit = inf.optsolve(problem, optimizer, diagnostics=False)
    np.testing.assert_allclose(fit.params, _EXACT, rtol=0.0, atol=_LSQ_ATOL)


def test_optsolve_defaults_to_the_exact_newton_trust_region():
    """optsolve without an optimizer runs the exact NewtonTrustRegion, which refuses a Hessian
    scale on its matrix-free subproblem."""
    target = jnp.asarray([1.0, -2.0])
    fit = inf.optsolve(
        inf.LeastSquaresProblem.from_residual(jnp.zeros(2), lambda x: x - target), diagnostics=False
    )
    assert fit.status is inf.TerminationStatus.GRAD_TOL and fit.info["hessian"] == "true"
    # measured 0.0 (one Newton step on a quadratic), CPU VM 2026-10-22,
    # pytest tests/contracts/test_structural_fitting.py
    np.testing.assert_allclose(fit.params, target, atol=1e-12)
    problem = inf.LeastSquaresProblem.from_residual(
        np.zeros(3), lambda x: jnp.asarray(_MATRIX) @ x - jnp.asarray(_TARGET)
    )
    with pytest.raises(ds.ConfigurationError, match="HessianScale.*matrix-free.*ConstantScale"):
        inf.NewtonTrustRegion(subproblem="cg").solve(problem)


def test_the_exact_newton_hessian_is_formed_in_tiles_above_the_dense_cutoff():
    """Above dense_cutoff=64 parameters the default hessian_tile forms the exact Hessian 32
    directions at a time, and the steps match those on the whole Hessian."""
    rng = np.random.default_rng(1)
    matrix, target = (
        jnp.asarray(rng.standard_normal((80, 70))),
        jnp.asarray(rng.standard_normal(80)),
    )
    problem = inf.LeastSquaresProblem.from_residual(np.zeros(70), lambda x: matrix @ x - target)
    tiled, whole = (
        inf.optsolve(problem, inf.NewtonTrustRegion(max_steps=3, **tile), diagnostics=False)
        for tile in ({}, {"hessian_tile": "full"})
    )
    assert (tiled.info["hessian_tile"], whole.info["hessian_tile"]) == (32, "full")
    # measured 5.0e-16 worst parameter difference, workstation CPU 2026-10-28,
    # pytest tests/contracts/test_structural_fitting.py
    np.testing.assert_allclose(tiled.params, whole.params, rtol=0.0, atol=1e-12)


def test_optsolve_fits_window_starts_with_gauss_newton_by_default():
    """optsolve without an optimizer runs NewtonTrustRegion(hessian="gauss-newton") when the
    multiple-shooting window starts are decision variables."""
    spec = ds.problem("structural_window_starts")
    spec.domain("line", ds.cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    rate = spec.param("rate", value=0.45, learnable=True, bounds=(0.1, 1.2))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
    spec.equation(writes="u", reads=("u",))(lambda q: -q.p.rate * q.u)
    times = jnp.linspace(0.0, 1.0, 5)
    measurement = inf.Measurement("u", times, jnp.exp(-0.6 * times)[:, None], weight=1e4)
    experiment = inf.Experiment(
        spec.build(), solver(), (0.0, 1.0), spec.assemble_state({}), (measurement,)
    )
    problem = inf.InferenceProblem(
        experiment, parameters=rate, trajectory=inf.Multishoot(inf.Windows.uniform(2))
    )
    fit = inf.optsolve(problem, diagnostics=False)
    assert problem.n_params > 1 and fit.info["hessian"] == "gauss-newton"


def test_newton_gauss_newton_records_the_final_residual_and_jacobian():
    """NewtonTrustRegion(hessian="gauss-newton") records the rows and Jacobian it holds at the
    returned point, whose cost they reproduce; the true-Hessian path records neither."""
    times = jnp.linspace(0.0, 3.0, 12)
    data = 2.0 * jnp.exp(-0.7 * times) + 0.01 * jnp.sin(5.0 * times)

    def residual(p):
        return p[0] * jnp.exp(-p[1] * times) - data

    problem = inf.LeastSquaresProblem.from_residual(np.asarray([1.0, 0.3]), residual)
    fit = inf.optsolve(problem, inf.NewtonTrustRegion(hessian="gauss-newton"), diagnostics=False)
    point = jnp.asarray(fit.params)
    # the fit ends STEP_TOL after 6 steps; measured 0.0 against a fresh evaluation at the
    # returned point, CPU VM 2026-10-22, pytest tests/contracts/test_structural_fitting.py
    np.testing.assert_allclose(fit.info["final_residual"], residual(point), atol=1e-12)
    # measured 0.0 on the Jacobian likewise, the same run
    np.testing.assert_allclose(fit.info["final_jacobian"], jax.jacfwd(residual)(point), atol=1e-12)
    assert 0.5 * float(jnp.sum(jnp.asarray(fit.info["final_residual"]) ** 2)) == fit.cost
    true = inf.optsolve(problem, inf.NewtonTrustRegion(hessian="true"), diagnostics=False)
    assert "final_residual" not in true.info and "final_jacobian" not in true.info


@pytest.fixture(scope="module")
def constrained():
    """min |x - (1, 1)|^2 subject to x0 <= 0.5, with rows weighted by 1e3."""
    target = jnp.asarray([1.0, 1.0])
    problem = inf.LeastSquaresProblem.from_residual(np.zeros(2), lambda x: 1.0e3 * (x - target))
    constraints = inf.InequalityConstraints(g=lambda x: jnp.atleast_1d(x[0] - 0.5), n_constraints=1)
    return inf.constrained_optsolve(
        problem, inf.LevenbergMarquardt(max_steps=60), constraints=constraints, outer_max=10
    )


def test_constrained_optsolve_meets_the_kkt_point_and_the_outer_stationarity_test(constrained):
    """The fit ends at x = (0.5, 1), multiplier 5e5, its inner solves passing the relative
    outer stationarity test after one penalty increase."""
    assert constrained.converged
    # measured 7.8e-7 on the point and 1.5e-6 relative on the multiplier, CPU VM 2026-10-22,
    # pytest tests/contracts/test_structural_fitting.py
    np.testing.assert_allclose(constrained.result.params, [0.5, 1.0], atol=1e-5)
    np.testing.assert_allclose(constrained.multipliers, [5.0e5], rtol=1e-5)
    assert constrained.rho == 1e7
    assert constrained.kkt["stationarity"] <= 1e-6 * constrained.kkt["stationarity_scale"]


def test_a_rejected_trial_predicate_costs_no_residual_evaluation():
    """LevenbergMarquardt asks the predicate before evaluating a trial's residual."""
    problem = inf.LeastSquaresProblem.from_residual(
        np.zeros(1),
        lambda x: x - 10.0,
        is_valid=lambda x: jnp.all(jnp.abs(x) <= 0.1),
    )
    fit = inf.LevenbergMarquardt(max_steps=1).solve(problem)
    assert fit.info["trial_predicate_rejections"] == 1
    assert fit.eval_counts.residual_evals == 1


def test_a_solver_whose_requirements_the_problem_lacks_is_refused_with_the_fitting_solvers_named():
    """optsolve refuses LevenbergMarquardt on a scalar-objective problem, naming the missing
    residual structure and the solvers whose requirements the problem meets."""
    problem = inf.OptimizationProblem.from_value(np.zeros(2), lambda x: jnp.sum((x - 1.0) ** 2))
    with pytest.raises(ds.ConfigurationError) as refusal:
        inf.optsolve(problem, inf.LevenbergMarquardt(), diagnostics=False)
    message = str(refusal.value)
    assert "residual_jacobian" in message and "LevenbergMarquardt requires" in message
    named = message.split("meets:")[1]
    assert all(name in named for name in ("BFGS", "GradientDescent", "Optax"))
    assert "NewtonTrustRegion" not in named and "TrustRegion" not in named


def test_the_problem_is_a_pytree_whose_value_and_gradient_composes_under_jit_and_vmap():
    """A least-squares problem's leaves are its arrays; its value and gradient under jax.jit
    (also with the problem as an argument) and jax.vmap equal the eager evaluation."""
    times = jnp.linspace(0.0, 2.0, 7)
    problem = inf.LeastSquaresProblem.from_residual(
        np.asarray([1.0, 0.5]), lambda p: p[0] * jnp.exp(-p[1] * times) - 0.8, bounds=(0.0, None)
    )
    assert all(isinstance(leaf, (np.ndarray, jax.Array)) for leaf in jax.tree.leaves(problem))
    assert len(jax.tree.leaves(problem)) == 3
    points = jnp.asarray([[1.0, 0.5], [0.7, 1.2], [2.0, 0.1]])
    eager = [problem.value_and_gradient(point) for point in points]
    jitted = [jax.jit(problem.value_and_gradient)(point) for point in points]
    passed = [jax.jit(lambda p, x: p.value_and_gradient(x))(problem, point) for point in points]
    values, gradients = jax.vmap(problem.value_and_gradient)(points)
    for index, (value, gradient) in enumerate(eager):
        for other in (jitted[index], passed[index]):
            np.testing.assert_array_equal(other[0], value)
            np.testing.assert_array_equal(other[1], gradient)
        # measured 0.0 between the batched and the eager evaluation, CPU VM 2026-10-22,
        # pytest tests/contracts/test_structural_fitting.py
        np.testing.assert_allclose(values[index], value, rtol=1e-15, atol=0.0)
        np.testing.assert_allclose(gradients[index], gradient, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=("float32", "float64"))
def test_value_and_jacfwd_is_jit_vmap_grad_and_dtype_safe(dtype):
    """value_and_jacfwd keeps the dtype, batches under vmap and differentiates under grad."""
    matrix = jnp.asarray([[2.0, 1.0], [-1.0, 3.0]], dtype=dtype)
    point = jnp.asarray([0.25, -0.5], dtype=dtype)
    value, jacobian = jax.jit(lambda x: value_and_jacfwd(lambda y: matrix @ y, x))(point)
    batched = jax.vmap(lambda x: value_and_jacfwd(lambda y: matrix @ y, x)[0])(
        jnp.stack((point, 2.0 * point))
    )
    gradient = jax.grad(lambda x: jnp.sum(value_and_jacfwd(lambda y: matrix @ y, x)[0] ** 2))(point)
    assert value.dtype == jacobian.dtype == dtype and batched.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    np.testing.assert_array_equal(jacobian, matrix)


def test_tiled_forward_jacobians_equal_the_dense_one():
    """value_and_jacfwd_tiled over tiles of 8 columns gives the dense Jacobian bit for bit."""
    point = jnp.linspace(-0.4, 0.8, 17)

    def function(x):
        return jnp.sin(x[:9]) + x[8:] ** 2

    dense_value, dense_jacobian = value_and_jacfwd(function, point)
    tiled_value, tiled_jacobian = value_and_jacfwd_tiled(function, point, tile=8)
    # measured 4.0e-16 relative between the separately compiled pushes, CPU VM 2026-10-22,
    # pytest tests/contracts/test_structural_fitting.py
    np.testing.assert_allclose(tiled_value, dense_value, rtol=8e-16, atol=0.0)
    np.testing.assert_array_equal(tiled_jacobian, dense_jacobian)


def test_a_forward_jacobian_integrates_its_solve_once():
    """value_and_jacfwd and linearize_value through an adaptive solve hold one solve loop each."""
    problem = ds.MassProblem(
        rhs=lambda t, y, theta, control: -theta * y,
        mass=ds.MassOperator.identity(2),
        structure=ds.DAEStructure(differential_vars=np.ones(2, dtype=bool)),
    )

    def final_state(p):
        return ds.pdesolve(problem, solver(), (0.0, 1.0), jnp.ones(2) * p[1], theta=p[0]).ys[-1]

    def linearized(p):
        value, pushforward = linearize_value(final_state, p)
        return value, jacfwd_from_pushforward(pushforward, p, value)

    point = jnp.asarray([0.4, 1.5])
    one_push = primitive_names(lambda p: jax.jvp(final_state, (p,), (p,)), point).count("while")
    assert (
        primitive_names(lambda p: value_and_jacfwd(final_state, p), point).count("while")
        == one_push
    )
    assert primitive_names(linearized, point).count("while") == one_push


def test_a_lumped_kl_eigsh_basis_is_byte_stable_for_identical_inputs():
    """KLBasis.from_lumped_covariance with eigsh is identical whatever the global NumPy seed."""
    points = np.linspace(0.0, 1.0, 12)
    covariance = np.exp(-(((points[:, None] - points[None, :]) / 0.2) ** 2))
    lumped = np.linspace(0.5, 1.5, points.size)
    # eigsh tolerance; measured byte-identical runs, CPU VM 2026-10-22
    np.random.seed(1)
    first = KLBasis.from_lumped_covariance(covariance, lumped, n_modes=4, solver="eigsh", tol=1e-12)
    np.random.seed(987654)  # measured byte-identical runs, the same tolerance
    second = KLBasis.from_lumped_covariance(
        covariance, lumped, n_modes=4, solver="eigsh", tol=1e-12
    )
    np.testing.assert_array_equal(first.modes, second.modes)
    np.testing.assert_array_equal(first.eigenvalues, second.eigenvalues)


def test_the_cost_threshold_is_cost_tol_times_the_cost_scale():
    """cost_threshold is cost_tol * max(|cost|, cost_atol / cost_tol), and a cost of 0 has a
    positive scale."""
    for cost in (0.0, -1e-30, 1e-12, 0.5, 3.0, 1e12):
        for cost_tol, cost_atol in ((1e-8, 0.0), (1e-8, 1e-10), (1e-6, 1e-3)):
            threshold = cost_threshold(cost, cost_tol, cost_atol)
            if threshold > 0.0:
                # measured 0.0 relative on every row, CPU VM 2026-10-22,
                # pytest tests/contracts/test_structural_fitting.py
                np.testing.assert_allclose(
                    cost_tol * cost_scale(cost, cost_tol, cost_atol), threshold, rtol=1e-14
                )
    assert cost_scale(0.0, 1e-8, 0.0) > 0.0
    # measured 1.4e-16 relative (the 1e-4 floor), CPU VM 2026-10-22,
    # pytest tests/contracts/test_structural_fitting.py
    np.testing.assert_allclose(
        cost_scale(np.asarray([-2.0, 1e-20, 5.0]), 1e-8, 1e-12), [2.0, 1e-4, 5.0], rtol=1e-15
    )


# Rejected trials below the last accepted radius whose cost rose: (trial cost over the cost,
# predicted decrease over the cost, scaled step norm).
_NOISY_TRIALS = ((1.0 + 3e-7, 2e-7, 1e-3), (1.0 + 1e-7, 5e-8, 4e-4), (1.0 + 2e-7, 1e-8, 2e-4))


def _loop_numbers(unit):
    """What the loop reads after three noise-dominated trials at a cost of ``2 * unit``."""
    cost, bound = 2.0 * unit, 1e-2 * unit
    trials = Trials(0, [], [], [], 1.0, 0, 0, [])
    for trial, predicted, norm in _NOISY_TRIALS:
        after = _after_attempt(
            trials,
            accepted=False,
            valid=True,
            invalid=False,
            cost=cost,
            trial_cost=trial * cost,
            predicted=predicted * cost,
            step_radius=0.5,
            cost_tol=1e-8,
            cost_atol=0.0,
            noise_limit=None,
            scaled_norm=norm,
            floor_bound=bound,
        )
        trials = after.trials
    relative_to = cost_scale(cost, 1e-8, 0.0)
    noise = float(np.median(trials.noise_dominated_cost_change))
    level = float(np.median(trials.gradient_floor_samples))
    sample = _floor_sample(
        evaluated=True,
        accepted=False,
        step_radius=0.5,
        accepted_radius=1.0,
        cost=cost,
        trial_cost=cost * (1.0 + 2e-9),
        predicted=cost * 1e-9,
        scaled_norm=1e-3,
        noise=noise,
        relative_to=relative_to,
        bound=bound,
    )
    # A trial whose cost rose by what the model predicted it would fall: the classical gain
    # ratio is -1, the ratio relaxed by the measured noise accepts it.
    rho, accepted, _radius, _reason = _judge(
        cost=cost,
        trial_cost=cost * (1.0 + 1e-9),
        predicted=cost * 1e-9,
        valid=True,
        invalid=False,
        scaled_norm=1e-3,
        radius=0.5,
        lock=0,
        noise_window=trials.noise_dominated_cost_change,
        eta=1e-4,
        radius_max=1000.0,
        relaxed=True,
        relative_to=relative_to,
    )
    return {
        "rejected": (trials.rejected_cost_change, trials.rejected_predicted_decrease),
        "noise": (trials.noise_dominated_cost_change, trials.consecutive_noise_dominated_trials),
        "floor": (level / unit, _floor_sigma(level, 0.01, relative_to), sample / unit),
        "verdict": (rho, accepted),
        "threshold": after.cost_threshold / unit,
    }


# Powers of two, so that the scaled costs carry the same rounding: costs from 1.8e-12 to 2.1e6.
@pytest.mark.parametrize("unit", [2.0**-40, 2.0**-20, 2.0**20])
def test_noise_windows_floor_and_gain_ratio_do_not_depend_on_cost_units(unit):
    """Three noise-dominated trials at a cost of 2 and of 2 * unit fill the same windows,
    measure the same noise, give the same floor samples and the same relaxed gain ratio."""
    reference = _loop_numbers(1.0)
    # measured at unit 1, CPU VM 2026-10-22: three noise-dominated trials, noise 2.0e-7
    # relative, floor level 1.0e-4 per unit scaled step, floor sigma 5.0e-7, relaxed gain
    # ratio 0.9975 (accepted); every unit agrees exactly.
    assert reference["noise"][1] == 3 and reference["verdict"][1]
    assert _loop_numbers(unit) == reference
