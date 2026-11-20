"""Linked experiments: a lane driven by another lane's solved series, with the derivative carried.

The source experiment produces an output (``inf.Output``) and two driven
experiments read it into a fixed signal of their model (``inf.Link``). The
lowered study solves the source first, then the driven experiments on its output;
the Gauss-Newton gradient carries the source's shared rate into the driven lanes
by the chain rule.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.optimization import inference as inf
from tests.support.links import (
    LINK_TIMES,
    START,
    TRUE_POINT,
    driven_alone,
    driven_experiment,
    linked_recipes,
    linked_study,
    source_experiment,
)

#: The point every derivative check is made at, away from the truth.
POINT = np.asarray([0.7, 0.5, 1.1, 1.0])


@pytest.fixture(scope="module")
def recipes():
    return linked_recipes("public_links")


@pytest.fixture(scope="module")
def lowered(recipes):
    """The linked study of exact data, lowered once."""
    study = ds.lower(linked_study(recipes))
    study.prepare(modes=("value", "blocks", "gradient", "frozen"))
    return study


def _five_point(values: np.ndarray, step: float) -> np.ndarray:
    """The five-point central difference from the costs at x + h, x - h, x + 2h, x - 2h per column."""
    n = values.shape[0] // 4
    plus, minus, plus2, minus2 = (values[k * n : (k + 1) * n] for k in range(4))
    return (8.0 * (plus - minus) - (plus2 - minus2)) / (12.0 * step)


def test_the_gauss_newton_gradient_of_a_linked_study_matches_central_differences(lowered):
    """The blocks gradient J^T r, with the source's rate carried into the driven lanes through
    the link, matches the five-point central difference of the value cost in every column."""
    _, gradient, _, failed = lowered.gauss_newton_fn(POINT)
    step = 4e-3
    eye = np.eye(POINT.size)
    shifts = np.concatenate([POINT + step * eye, POINT - step * eye])
    shifts = np.concatenate([shifts, POINT + 2 * step * eye, POINT - 2 * step * eye])
    reference = _five_point(lowered.value_and_gradient(shifts).values, step)
    assert failed == 0
    # measured 1.3e-7 relative (column b of c2), CPU 2026-11-20, pytest
    # tests/public_api/test_links.py
    np.testing.assert_allclose(np.asarray(gradient), reference, rtol=1e-6, atol=1e-6)


def test_the_gradient_mode_equals_the_blocks_gradient(lowered):
    """value_and_gradient (the gradient mode) carries the link as the blocks mode does."""
    _, blocks, _, _ = lowered.gauss_newton_fn(POINT)
    gradient = lowered.value_and_gradient(POINT).gradients[0]
    np.testing.assert_array_equal(np.asarray(gradient), np.asarray(blocks))


def test_a_driven_lane_equals_a_separate_solve_on_the_source_output(recipes, lowered):
    """The driven lane's cost and b-gradient at a point equal those of the driven experiment
    alone, its signal built with the source's output at that point. They agree to the solver's
    tolerance: the linked lane also carries the source's rate as a tangent, which the step
    control of its forward-mode solve sees."""
    results = lowered.value_and_gradient(POINT).results
    values = np.asarray(results.outputs["source"][0][0])
    position = results.lane_ids.index("c1")
    model = driven_alone(recipes, values)
    alone = driven_experiment(recipes, "c1", model=model, links=(), parameters=None)
    study = ds.lower(inf.Study((alone,), parameters=ds.ParameterSet(shared=(recipes.a, recipes.b))))
    study.prepare(modes=("gradient",))
    other = study.value_and_gradient(POINT[[0, 2]]).results
    assert values.shape == LINK_TIMES.shape
    linked = (results.cost[0, position], results.gradient[0, position, 1])
    # measured 1.3e-10 relative (the solver's rtol is 1e-10), CPU 2026-11-20, pytest
    # tests/public_api/test_links.py
    np.testing.assert_allclose(linked, (other.cost[0, 0], other.gradient[0, 0, 1]), rtol=1e-8)


@pytest.mark.parametrize("nested_at", [None, "cell"], ids=["joint", "nested"])
def test_trust_region_fits_recover_the_source_and_driven_rates(lowered, nested_at):
    """From a wrong start, the joint and the nested TrustRegion fits of exact data recover the
    source's rate and each cell's rate; the source cell's inert rate stays where it started.
    The nested fit's polish steps on the frozen Jacobians of the cells' lanes, its source solved
    again at every request of a cell's lanes."""
    fit = inf.optsolve(
        lowered.with_start(START),
        inf.TrustRegion(max_steps=40, nested_at=nested_at),
        diagnostics=False,
    )
    # measured 2.7e-9 relative (nested, b of c1), CPU 2026-11-20, pytest
    # tests/public_api/test_links.py
    np.testing.assert_allclose(np.asarray(fit.params), TRUE_POINT, rtol=1e-6)


def test_a_failed_source_fails_the_lanes_it_drives(recipes):
    """A source whose output is not finite at a point fails there, and so does every lane it
    drives, with the source's kind; at another point every lane solves."""
    study = ds.lower(linked_study(recipes, observe=lambda state: jnp.log(jnp.mean(state) - 0.3)))
    study.prepare(modes=("gradient",))
    solved = study.value_and_gradient(np.asarray([0.3, 0.5, 1.1, 1.0]))
    failed = study.value_and_gradient(POINT)
    assert not solved.failures
    assert {(entry["lane"], entry["kind"]) for entry in failed.failures} == {
        ("source", "not_finite"),
        ("c1", "not_finite"),
        ("c2", "not_finite"),
    }


def _with(recipes, change):
    """The study with one experiment changed by ``change(experiments)``."""
    experiments = list(linked_study(recipes).experiments)
    change(experiments)
    hierarchy = inf.Hierarchy(levels=("cell",), shared=(recipes.a,), per={"cell": (recipes.b,)})
    return inf.Study(tuple(experiments), parameters=hierarchy)


def _relink(index, **link):
    def change(experiments):
        given = {"source": "source", "output": "y", "into": "y"} | link
        experiments[index] = dataclasses.replace(experiments[index], links=(inf.Link(**given),))

    return change


def _source_output(times):
    def change(experiments):
        output = inf.Output("y", times=times, observe=lambda state: jnp.mean(state))
        experiments[0] = dataclasses.replace(experiments[0], outputs=(output,))

    return change


def _chained(experiments):
    output = inf.Output("w", times=LINK_TIMES, observe=lambda state: jnp.mean(state))
    experiments[1] = dataclasses.replace(experiments[1], outputs=(output,))
    experiments[2] = dataclasses.replace(
        experiments[2], links=(inf.Link(source="c1", output="w", into="y"),)
    )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (_relink(1, source="nowhere"), "which is not in the study"),
        (_relink(1, output="v"), r"declares the outputs \['y'\]"),
        (_relink(1, source="c1"), "to its own output"),
        (_source_output(LINK_TIMES[:5]), "has 5 times"),
        (_chained, "links have one level"),
    ],
    ids=["unknown-source", "unknown-output", "own-output", "values-length", "two-levels"],
)
def test_a_study_refuses_links_that_do_not_fit_its_experiments(recipes, change, match):
    """Study refuses a link to an unknown experiment or output, to its own output, into a signal
    whose values do not have one entry per output time, and a driven experiment that is a
    source itself."""
    with pytest.raises(ValueError, match=match):
        _with(recipes, change)


@pytest.mark.parametrize(
    ("link", "match"),
    [
        (inf.Link(source="source", output="y", into="nope"), "names no function of the model"),
        (inf.Link(source="source", output="y", into="b"), "names no function of the model"),
    ],
    ids=["unknown-function", "a-parameter"],
)
def test_an_experiment_refuses_a_link_into_anything_but_a_fixed_signal(recipes, link, match):
    """Experiment refuses a link whose into= names no function of its model."""
    with pytest.raises(ValueError, match=match):
        driven_experiment(recipes, "c1", links=(link,))


def test_an_experiment_refuses_a_link_into_a_learnable_function(recipes):
    """A learnable signal cannot be filled by a link."""
    spec = ds.problem("public_links_learnable")
    spec.domain("line", ds.domains.cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    spec.param("a", value=0.5, learnable=True)
    spec.param("b", value=0.5, learnable=True)
    signal = inf.approx.PiecewiseLinear(jnp.asarray(LINK_TIMES), jnp.zeros(LINK_TIMES.shape))
    spec.function("y", signal, learnable=True)
    spec.field("z", mass="differential", scale=1.0, init=lambda x: jnp.zeros_like(x))

    @spec.equation(writes="z", reads=("z",))
    def relax(q):
        return -q.p.b * q.z + q.fn.y(q.t)

    with pytest.raises(ValueError, match="names a learnable function"):
        driven_experiment(recipes, "c1", model=spec.build())


def test_an_experiment_refuses_outputs_and_links_on_windows(recipes):
    """Outputs and links are read from one solve over t_span; a Multishoot is refused."""
    windows = inf.Multishoot(inf.Windows.uniform(2))
    with pytest.raises(ValueError, match="splits the span into windows"):
        source_experiment(recipes, trajectory=windows)


def test_a_linked_study_is_fitted_on_its_lanes_only(recipes, lowered):
    """An InferenceProblem of a linked study, and a fit of the lowered study by an optimizer of
    the joint residual, are refused: the joint residual solves each experiment alone."""
    with pytest.raises(ValueError, match="lower the study with ds.lower"):
        inf.InferenceProblem(linked_study(recipes))
    with pytest.raises(ValueError, match="Fit it on its lanes"):
        inf.optsolve(lowered, inf.LevenbergMarquardt(max_steps=2), diagnostics=False)
