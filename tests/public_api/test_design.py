"""Design: Fisher information, optimality criteria, sensitivity solves, robust and sequential designs."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup import inference as inf
from diff_soup.domains import cartesian_grid_1d
from diff_soup.inference import design
from tests.support.inference import decay_experiment, decay_recipe, solver

_SENSITIVITIES = jnp.asarray(
    [[[1.0, 0.2, 0.0]], [[0.5, 1.0, 0.1]], [[0.0, 0.3, 2.0]], [[0.4, 0.0, 0.5]]]
)  # (frames, channels, directions)


def test_fisher_information_and_optimality_criteria_follow_their_closed_forms():
    """The FIM sums s s^T; D, Ds and smooth E optimality follow their closed forms."""
    fim = design.fisher_information(_SENSITIVITIES)
    flat = np.asarray(_SENSITIVITIES).reshape(-1, 3)
    logdet = np.linalg.slogdet(fim)[1]
    # measured 0.0 and 0.0 differences, CPU VM 2026-10-22, pytest tests/public_api/test_design.py
    np.testing.assert_allclose(fim, flat.T @ flat, rtol=1e-14)
    np.testing.assert_allclose(design.d_optimal(fim, ridge=0.0), logdet, rtol=1e-12)
    dense = np.asarray(fim)
    schur = dense[:1, :1] - dense[:1, 1:] @ np.linalg.solve(dense[1:, 1:], dense[1:, :1])
    # measured 5.3e-15 relative, CPU VM 2026-10-22, pytest tests/public_api/test_design.py
    np.testing.assert_allclose(
        design.ds_optimal(fim, (0,), ridge=0.0), np.log(schur[0, 0]), rtol=1e-12
    )
    assert float(design.e_optimal_smooth(fim, ridge=0.0)) <= np.log(np.linalg.eigvalsh(dense)[0])
    with pytest.raises(ValueError):
        design.ds_optimal(fim, (0, 1, 2))


def test_fisher_information_of_a_linear_model_with_diagonal_noise_is_a_t_sigma_inverse_a():
    """For y = A theta with independent noise sigma, the FIM is A^T Sigma^-1 A."""
    sigma = jnp.asarray([[0.5], [2.0], [1.0], [0.25]])  # (frames, channels)
    fim = design.fisher_information(_SENSITIVITIES, sigma)
    rows = np.asarray(_SENSITIVITIES).reshape(-1, 3)
    expected = rows.T @ np.diag(1.0 / np.ravel(sigma) ** 2) @ rows
    # measured 0.0 difference, CPU VM 2026-10-22, pytest tests/public_api/test_design.py
    np.testing.assert_allclose(fim, expected, rtol=1e-14)


def _decay_with_design(name):
    spec = ds.problem(name)
    spec.domain("line", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    rate = spec.param("rate", value=0.4, learnable=True)
    gain = spec.param("design", value=1.2, learnable=True)
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))

    @spec.equation(writes="u", reads=("u",))
    def decay(q):
        return -(q.p.rate * q.p.design) * q.u

    return design.sensitivity_solve(
        spec.build(),
        (0.0, 0.1),
        spec.assemble_state({}),
        wrt=(rate,),
        design=(gain,),
        saveat=ds.SaveAt(t1=True),
        solver=solver(),
        consistent_ic=False,
        max_steps=100,
    )


@pytest.fixture(scope="module")
def run():
    return _decay_with_design("design_sensitivity_decay")


def test_a_sensitivity_solve_matches_the_closed_form_sensitivity(run):
    """d u(0.1) / d rate of u' = -rate * gain * u is -0.1 gain exp(-0.48 * 0.1)."""
    sensitivity = jax.jacfwd(run)(run.x0, run.design_theta(run.d0))
    exact = -0.1 * 1.2 * np.exp(-0.4 * 1.2 * 0.1)
    # measured 1.4e-5 relative at solver rtol 1e-6, CPU VM 2026-10-22,
    # pytest tests/public_api/test_design.py
    np.testing.assert_allclose(np.ravel(sensitivity), [exact], rtol=1e-4)


def test_the_reverse_over_forward_design_gradient_is_finite_and_nonzero(run):
    """jax.grad over a jacfwd of the sensitivity run moves with the design parameter."""

    def criterion(point):
        return jnp.sum(jax.jacfwd(run)(run.x0, run.design_theta(point)) ** 2)

    gradient = jax.grad(criterion)(run.d0)
    assert gradient.shape == (1,) and bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.abs(gradient[0])) > 0.0


def test_the_sharded_robust_gradient_equals_the_mean_criterion_on_one_device():
    """sharded_robust_gradient returns the value and gradient of robust_criterion."""
    samples = {"rate": jnp.asarray([0.2, 0.4, 0.8, 1.3]), "gain": jnp.arange(4.0) + 1}
    point = jnp.asarray([0.9, 1.2])

    def criterion(sample, at):
        return jnp.log1p(jnp.sum((sample["gain"] * jnp.exp(-sample["rate"] * at)) ** 2))

    value, gradient = design.sharded_robust_gradient(criterion, point, samples)
    expected = jax.value_and_grad(lambda at: design.robust_criterion(criterion, samples, at))(point)
    # measured 0.0 and 0.0 relative, CPU VM 2026-10-22, pytest tests/public_api/test_design.py
    np.testing.assert_allclose(np.asarray(value), np.asarray(expected[0]), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(gradient), np.asarray(expected[1]), rtol=1e-12)


@pytest.fixture(scope="module")
def sequential():
    """A decay problem's expected-curvature posterior and two candidates of 1x and 2x its data."""
    decay = decay_recipe("design_sequential_decay")
    problem = inf.InferenceProblem(decay_experiment(decay), parameters=decay.rate)
    posterior = inf.laplace(problem, {decay.rate: 0.6}, curvature="expected")

    def candidate(name, repeats):
        def information_at(mode):
            fresh = inf.information(problem, at=mode, kind="expected")
            provenance = {**fresh.provenance, "linearized_at": tuple(map(float, np.asarray(mode)))}
            return dataclasses.replace(
                fresh, expected=repeats * fresh.expected, provenance=provenance
            )

        return design.CandidateDesign(
            name, repeats, information_at, frozenset({f"{name}-y"}), frozenset()
        )

    space = design.ExperimentSpace((candidate("once", 1.0), candidate("twice", 2.0)))
    state = design.SequentialState(posterior, frozenset(), frozenset())
    return posterior, space, state


def test_sequential_step_selects_the_most_informative_candidate_and_replays(sequential):
    """sequential_step picks the larger information gain; replaying its update advances once."""
    posterior, space, state = sequential
    choice = design.sequential_step(state, space)
    update = design.SequentialUpdate(choice, posterior, choice.data_ids)
    replayed = design.replay_sequential(state, (update,))
    assert choice.design_id == "twice" and choice.score > 0.0
    assert replayed.iteration == 1 and replayed.assimilated_data_ids == {"twice-y"}
    with pytest.raises(ValueError, match="already assimilated"):
        design.sequential_update(replayed, update)


def test_candidate_designs_and_experiment_spaces_refuse_malformed_declarations():
    """A candidate needs response IDs apart from its conditioning IDs; a space unique IDs."""
    with pytest.raises(ValueError, match="data_ids must name"):
        design.CandidateDesign("a", None, lambda mode: None, frozenset(), frozenset())
    with pytest.raises(ValueError, match="disjoint"):
        design.CandidateDesign("a", None, lambda mode: None, frozenset({"y"}), frozenset({"y"}))
    one = design.CandidateDesign("a", None, lambda mode: None, frozenset({"y"}), frozenset())
    with pytest.raises(ValueError, match="unique"):
        design.ExperimentSpace((one, one))
    with pytest.raises(ValueError, match="at least one"):
        design.ExperimentSpace(())


_TASKS = tuple({"task_id": name, "value": value} for name, value in (("a", 1), ("b", 2), ("c", 3)))


def _campaign(path, ran, failing=(), tasks=_TASKS, **keywords):
    """checkpointed_map of ``10 * value`` over ``tasks``, recording each task it runs in ``ran``
    and raising for the task IDs in ``failing``."""

    def task(item):
        ran.append(item["task_id"])
        if item["task_id"] in failing:
            raise RuntimeError(f"task {item['task_id']} failed")
        return {"result": 10 * item["value"]}

    return design.checkpointed_map(task, tasks, ledger_path=path, **keywords)


def test_a_resumed_campaign_retries_only_its_failed_task_and_ends_ok(tmp_path):
    """A campaign whose task ``b`` failed keeps the failure in its ledger; resuming it runs
    ``b`` alone as attempt 2, and the resumed result is ok with every output in input order."""
    path, ran = tmp_path / "ledger.json", []
    first = _campaign(path, ran, failing=("b",))
    assert not first.ok and [failure.task_id for failure in first.failures] == ["b"]
    assert first.failures[0].error_type == "RuntimeError" and path.exists()
    resumed = _campaign(path, ran)
    assert ran == ["a", "b", "c", "b"]
    assert resumed.ok and [record.attempt for record in resumed.records] == [1, 2, 1]
    assert list(resumed) == [{"result": 10}, {"result": 20}, {"result": 30}]


def test_a_redeclared_task_id_is_refused_unless_the_campaign_restarts(tmp_path):
    """Resuming a ledger with a task ID whose declaration changed is refused, naming
    on_conflict; on_conflict="restart" discards the ledger and runs every task from attempt 1."""
    path = tmp_path / "ledger.json"
    _campaign(path, [])
    changed = (*_TASKS[:2], {"task_id": "c", "value": 4})
    with pytest.raises(ValueError, match="on_conflict='restart'"):
        _campaign(path, [], tasks=changed)
    ran = []
    restarted = _campaign(path, ran, tasks=changed, on_conflict="restart")
    assert ran == ["a", "b", "c"] and restarted.ok
    assert [record.attempt for record in restarted.records] == [1, 1, 1]
    assert restarted[2] == {"result": 40}
