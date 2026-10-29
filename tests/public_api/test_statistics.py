"""Statistics verbs: information, Laplace, profile, identifiability, reports and bootstrap."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup import inference as inf
from diff_soup.domains import cartesian_grid_1d
from diff_soup.inference.approx import legendre_weights
from diff_soup.inference.statistics import IdentifiedSubspace, ImproperPosteriorError
from tests.support.inference import (
    TIMES,
    decay_recipe,
    decay_study,
    decay_values,
    mean_measurement,
    small_fit,
    solver,
)


def _counted_decay(counter, *, likelihood=True):
    """A one-parameter decay whose right-hand side counts its executions in ``counter``."""
    spec = ds.problem("statistics_counted_decay")
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    rate = spec.param(
        "rate", value=0.45, learnable=True, prior=inf.NormalPrior(mean=0.5, scale=0.3)
    )
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))

    @spec.equation(writes="u", reads=("u",))
    def decay(q):
        jax.debug.callback(lambda: counter.append(None))
        return -q.p.rate * q.u

    noisy = decay_values() + np.asarray([0.0, 0.01, -0.01, 0.005, -0.005])
    measurement = (
        mean_measurement(noisy, likelihood=inf.Gaussian(scale=0.02))
        if likelihood
        else mean_measurement(noisy, weight=1.0)
    )
    experiment = inf.Experiment(
        model=spec.build(),
        solver=solver(),
        t_span=(0.0, 1.0),
        initial_state=spec.assemble_state({}),
        measurements=(measurement,),
        id="decay",
    )
    return experiment, rate


@pytest.fixture(scope="module")
def single():
    """A MAP fit of a one-rate decay, with the count of right-hand side runs it took."""
    counter = []
    experiment, rate = _counted_decay(counter)
    problem = inf.InferenceProblem(experiment, parameters=rate, objective=inf.MAP())
    fit = small_fit(problem, inf.LevenbergMarquardt(max_steps=30))
    jax.effects_barrier()
    return problem, rate, fit, counter


def test_information_and_expected_laplace_at_a_fit_reuse_its_jacobian(single):
    """Expected information and the expected-curvature Laplace at a fit run no new solve."""
    problem, _, fit, counter = single
    jax.effects_barrier()
    before = len(counter)
    inf.information(problem, at=fit, kind="expected")
    inf.laplace(problem, fit, curvature="expected")
    jax.effects_barrier()
    assert len(counter) == before


def test_laplace_defaults_to_observed_curvature_and_labels_it(single):
    """laplace() defaults to the observed curvature, labels it, and that curvature is symmetric."""
    problem, rate, fit, _ = single
    posterior = inf.laplace(problem, fit)
    expected = inf.laplace(problem, fit, curvature="expected")
    assert posterior.curvature_kind == "observed" and expected.curvature_kind == "expected"
    assert posterior.names == ("rate",) and float(posterior[rate].mean) == float(fit[rate])
    lower, upper = posterior[rate].interval()
    assert lower < float(fit[rate]) < upper
    observed = np.asarray(inf.information(problem, at=fit, kind="observed").observed)
    np.testing.assert_array_equal(observed, observed.T)


# measured 0.0 relative difference, workstation CPU 2026-10-28,
# pytest tests/public_api/test_statistics.py::test_the_laplace_evidence_of_a_map_fit_is_its_closed_form
_EVIDENCE = 1e-12
# measured 1.5e-8 relative difference between the float32 and float64 posteriors of the
# float32 study, workstation CPU 2026-10-28, bugfix/fpass/t_probe.py
_FLOAT32_EVIDENCE = 1e-6


def test_the_laplace_evidence_of_a_map_fit_is_its_closed_form(single):
    """evidence(problem, fit) and laplace(problem, fit).log_evidence equal
    log p(y | mode) + log p(mode) + d/2 log 2 pi - 1/2 log det H, with H the observed
    information plus the prior precision, so the user restates no prior. The log posterior
    of an ExperimentSet of the experiment is log p(y | mode) + log p(mode) at the mode."""
    problem, _, fit, _ = single
    experiments = inf.statistics.ExperimentSet(problem.study.experiments, independent=True)
    prior = inf.statistics.GaussianPrior(mean=jnp.array([0.5]), scale=jnp.array([0.3]))
    # 1.0.0 raised AttributeError ('observed'), then TypeError ('coordinate'); measured 0.0
    # relative difference after the fix, CPU workstation 2026-10-28, bugfix/we1b/r37.py
    value = float(experiments.log_posterior(prior=prior)(jnp.asarray(fit.params)))
    np.testing.assert_allclose(value, fit.log_likelihood + fit.log_prior, rtol=_EVIDENCE)
    curvature = inf.information(problem, at=fit)
    precision = np.asarray(curvature.observed) + np.asarray(curvature.prior_precision)
    closed = (
        fit.log_likelihood
        + fit.log_prior
        + 0.5 * precision.shape[0] * np.log(2.0 * np.pi)
        - 0.5 * np.linalg.slogdet(precision)[1]
    )
    values = [inf.evidence(problem, fit).log_evidence, inf.laplace(problem, fit).log_evidence]
    np.testing.assert_allclose(values, closed, rtol=_EVIDENCE)


def test_a_float32_fit_has_an_evidence_and_its_solve_sets_the_stationarity_bound():
    """A float32 study has an evidence at a float32 point equal to the one at its fit; a point
    off the mode by a decrement its float32 solve tolerates has an evidence, and a point
    farther off is refused, naming the solver tolerance and dtype."""
    # Measured decrements, workstation CPU 2026-10-28, bugfix/fpass/t_probe.py: 0.221 at the
    # rate + 0.003 against the bound 0.4 of the float32 solve, 3.70 at + 0.05.
    decay = decay_recipe(
        "statistics_float32",
        source=True,
        rate_prior=inf.NormalPrior(mean=0.5, scale=0.3),
        source_prior=inf.NormalPrior(mean=0.0, scale=0.1),
        precision=ds.Precision.fp32(),
    )
    study = inf.InferenceProblem(decay_study(decay), objective=inf.MAP())
    fit32 = small_fit(study, inf.LevenbergMarquardt(max_steps=30))
    narrow = inf.evidence(study, np.asarray(fit32.params, np.float32)).log_evidence
    at_fit = inf.evidence(study, fit32).log_evidence
    np.testing.assert_allclose(narrow, at_fit, rtol=_FLOAT32_EVIDENCE)
    rate = np.eye(4)[0]  # the shared rate leads the flat parameter order
    assert np.isfinite(inf.evidence(study, fit32.params + 0.003 * rate).log_evidence)
    with pytest.raises(ValueError, match="rtol 1e-06 in float32.*tighter solver tolerance"):
        inf.evidence(study, fit32.params + 0.05 * rate)


# measured 3.3e-3 relative difference (Continuity weight 1e6), workstation CPU 2026-10-28,
# pytest tests/public_api/test_statistics.py::test_laplace_on_a_multishoot_fit_integrates_out_the_window_starts
_MARGINAL_SD = 1e-2


def test_laplace_on_a_multishoot_fit_integrates_out_the_window_starts(single):
    """On a multiple-shooting fit, laplace covers the rate alone, the window starts integrated
    out, and its standard deviation is the single-shooting fit's."""
    problem, rate, fit, _ = single
    windowed = inf.InferenceProblem(
        problem.study.experiments[0],
        parameters=rate,
        objective=inf.MAP(),
        trajectory=inf.Multishoot(inf.Windows.at(TIMES), inf.Continuity(weight=1.0e6)),
    )
    posterior = inf.laplace(windowed, small_fit(windowed, inf.LevenbergMarquardt(max_steps=30)))
    assert posterior.names == ("rate",)
    single_sd = float(inf.laplace(problem, fit)[rate].sd)
    np.testing.assert_allclose(float(posterior[rate].sd), single_sd, rtol=_MARGINAL_SD)


def test_identifiability_and_profile_name_the_fitted_handle(single):
    """identifiability ranks the rate; profile ends each wing with a documented reason."""
    problem, rate, fit, _ = single
    report = inf.identifiability(problem, fit, parameter_scales={rate: 0.2})
    assert report.names == ("rate",) and report.raw.sensitivity_rank == 1
    curve = inf.profile(problem, fit, rate, bounds=(0.3, 0.9), max_steps=3)
    reasons = {"crossed", "bounds", "max_steps", "nonconvergence", "solver_failure"}
    assert curve.target == "rate" and set(curve.wing_end) == {"lower", "upper"}
    assert set(curve.wing_end.values()) <= reasons


def test_the_reports_read_the_fit_channel_by_channel(single):
    """residual_report, practical_report and function_report describe the same fit."""
    problem, _, fit, _ = single
    residuals = inf.residual_report(problem, fit)
    practical = inf.practical_report(problem, fit, posterior=inf.laplace(problem, fit))
    information = inf.information(problem, at=fit, kind="expected")
    assert set(residuals.channels) == {"mean"} and practical.model_adequacy is not None
    assert inf.function_report(information, "rate", gram=np.eye(1)).names == ("rate",)


def test_statistical_verbs_require_a_measurement_likelihood():
    """A weighted least-squares problem fits, but information names the measurement."""
    experiment, rate = _counted_decay([], likelihood=False)
    problem = inf.InferenceProblem(experiment, parameters=rate)
    with pytest.raises(ValueError, match="'mean'"):
        inf.information(problem, at={rate: 0.6})
    with pytest.raises(ValueError, match="'mean'"):
        inf.InferenceProblem(experiment, parameters=rate, objective=inf.MAP())


@pytest.mark.parametrize("count", (3, 5))
def test_tikhonov_default_weights_cover_odd_legendre_coefficient_counts(count):
    """The default Tikhonov rows carry the Legendre weights 1 + n(n+1) for any count."""
    weights = legendre_weights(count)
    rows = inf.Tikhonov(lambda_=0.5).residual(jnp.ones(count))
    assert weights.shape == rows.shape == (count,)
    np.testing.assert_array_equal(weights, 1.0 + np.arange(count) * (np.arange(count) + 1))


@pytest.fixture(scope="module")
def clustered():
    """A TrustRegion fit of the three-owner decay study, lowered."""
    decay = decay_recipe("statistics_clustered", source=True)
    lowered = ds.lower(decay_study(decay))
    return decay, lowered, small_fit(lowered, inf.TrustRegion(max_steps=30))


def test_cluster_covariance_counts_the_owners_of_a_lowered_fit(clustered):
    """cluster_covariance(lowered, fit, cluster="cell") clusters over the three owners."""
    _, lowered, fit = clustered
    robust = inf.cluster_covariance(lowered, fit, cluster="cell")
    assert robust.n_clusters == 3 and robust.covariance.shape == (4, 4)
    assert np.all(np.isfinite(robust.covariance))


@pytest.fixture(scope="module")
def bootstrapped(clustered):
    _, lowered, fit = clustered
    return inf.bootstrap(lowered, fit, replicates=2, key=jax.random.key(12), steps=1, chunk=None)


def test_a_one_step_bootstrap_keeps_finite_replicates_with_intervals_and_bands(
    clustered, bootstrapped
):
    """bootstrap(steps=1) keeps both replicates, finite, with a usable interval and band."""
    decay, _, _ = clustered
    interval = bootstrapped.interval(decay.rate)
    assert bootstrapped.n_kept == 2 and np.all(np.isfinite(bootstrapped.values(decay.rate)))
    assert isinstance(interval, inf.BootstrapInterval) and interval.status in {"ok", "unresolved"}
    grid = np.linspace(0.0, 1.0, 3)
    band = bootstrapped.band(lambda view, at: np.exp(-view[decay.rate] * at), at=grid)
    assert isinstance(band, inf.BootstrapBand) and len(band.status) == grid.size


# measured 0.0 from (1, 1) / sqrt(2) and (-1, 1) / sqrt(2) up to sign, workstation 2026-10-23,
# pytest tests/public_api/test_statistics.py::test_a_non_identifiable_study_raises_or_returns_its_identified_subspace
_BASIS = 1e-12


def test_a_non_identifiable_study_raises_or_returns_its_identified_subspace():
    """With rates ``a`` and ``b`` entering only as ``a + b`` and no prior, laplace raises
    ImproperPosteriorError under on_singular="raise" and returns the IdentifiedSubspace under
    "subspace": both name ``a + b`` as the one determined direction and ``a - b`` as null."""
    spec = ds.problem("statistics_summed_rates")
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    first = spec.param("a", value=0.3, learnable=True)
    second = spec.param("b", value=0.3, learnable=True)
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
    spec.equation(writes="u", reads=("u",))(lambda q: -(q.p.a + q.p.b) * q.u)
    experiment = inf.Experiment(
        model=spec.build(),
        solver=solver(),
        t_span=(0.0, 1.0),
        initial_state=spec.assemble_state({}),
        measurements=(mean_measurement(decay_values()),),
        id="summed",
    )
    problem = inf.InferenceProblem(experiment, parameters=(first, second))
    point = {first: 0.3, second: 0.3}
    with pytest.raises(ImproperPosteriorError) as raised:
        inf.laplace(problem, point)
    subspace = inf.laplace(problem, point, on_singular="subspace")
    assert isinstance(subspace, IdentifiedSubspace) and subspace.names == ("a", "b")
    np.testing.assert_array_equal(raised.value.subspace.directions, subspace.directions)
    np.testing.assert_allclose(np.abs(subspace.directions[:, 0]), np.sqrt(0.5), atol=_BASIS)
    np.testing.assert_allclose(
        subspace.null_directions[:, 0] * np.sign(subspace.null_directions[1, 0]),
        [-np.sqrt(0.5), np.sqrt(0.5)],
        atol=_BASIS,
    )


def test_profile_of_an_owner_varying_handle_walks_that_owners_value(clustered):
    """profile(owner=) of the per-owner source walks from that owner's fitted value and labels
    the curve with the owner; without owner= it is refused, naming the owners."""
    decay, lowered, fit = clustered
    curve = inf.profile(
        lowered,
        fit,
        decay.source,
        owner="cell-b",
        optimizer=inf.TrustRegion(max_steps=10),
        max_steps=1,
    )
    assert curve.target == "source@cell-b"
    assert float(fit[decay.source]["cell-b"]) in np.asarray(curve.grid).tolist()
    with pytest.raises(ValueError, match="cell-a.*cell-b.*cell-c"):
        inf.profile(lowered, fit, decay.source)
