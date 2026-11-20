"""Studies whose experiments read the handles they declare: ``inf.Study(reads="declared")``.

The study of ``tests.support.inference.reads_study`` holds three experiments; the third
one's model does not declare ``source``. Its twin declares ``source`` in that model and
never reads it, an inert placeholder, under ``reads="all"``. The two have the same fitted
vector, priors and data, so the lanes' blocks, the fits and the Laplace standard
deviations agree, while the third lane of the first is differentiated against ``rate``
alone.
"""

from __future__ import annotations

import numpy as np
import pytest

import diff_soup as ds
from diff_soup.optimization import inference as inf
from diff_soup.optimization.inference.engine._request import Request
from tests.support.inference import decay_experiment, decay_recipe, reads_study, small_fit

#: measured 5.6e-16 relative on the lanes' blocks, 6.8e-13 on the fitted values, 6.5e-13 on
#: the cost, 3.9e-12 on the Laplace sd (shared placement), CPU workstation 2026-11-20,
#: pytest tests/contracts/test_study_reads.py
AGREEMENT = 1e-10


@pytest.fixture(scope="module", params=("shared", "level"))
def pair(request):
    """``(declared, placeholder)``: the study of partial reads and its twin, lowered for a MAP fit."""
    declared = ds.lower(reads_study("declared", request.param), objective=inf.MAP())
    placeholder = ds.lower(reads_study("all", request.param), objective=inf.MAP())
    return declared, placeholder


def _close(actual, expected):
    """``actual`` equals ``expected`` to ``AGREEMENT`` relative to the largest entry."""
    expected = np.asarray(expected)
    scale = float(np.max(np.abs(expected))) or 1.0
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=AGREEMENT * scale)


def _widths(lowered) -> dict:
    return {lane: bucket.width for bucket in lowered.plan().buckets for lane in bucket.lanes}


def test_each_lane_is_differentiated_against_the_handles_its_model_declares(pair):
    """The third lane's narrow point holds ``rate`` alone; the placeholder's holds both."""
    declared, placeholder = pair
    assert _widths(declared) == {"e1": 2, "e2": 2, "e3": 1}
    assert _widths(placeholder) == {"e1": 2, "e2": 2, "e3": 2}
    np.testing.assert_array_equal(declared.x0, placeholder.x0)


def test_the_lanes_blocks_equal_those_of_the_placeholder_study(pair):
    """Every lane's cost, ``J^T r`` and ``J^T J`` in its full point equal its twin's."""
    point = np.asarray(pair[0].x0) + np.linspace(0.01, 0.02, np.size(pair[0].x0))
    results = []
    for lowered in pair:
        lowered.prepare(modes=("blocks",))
        results.append(lowered.evaluate(Request(mode="blocks", points=point)))
    declared, placeholder = results
    assert declared.lane_ids == placeholder.lane_ids
    for field in ("cost", "gradient", "normal"):
        _close(getattr(declared, field), getattr(placeholder, field))


@pytest.mark.parametrize("nested_at", [None, "cell"], ids=("joint", "nested"))
def test_the_fit_and_its_laplace_sd_equal_those_of_the_placeholder_study(pair, nested_at):
    """The trust region, joint or nested at ``cell``, reaches the twin's optimum, cost and sd."""
    options = {} if nested_at is None else {"nested_at": nested_at}
    fits = [small_fit(lowered, inf.TrustRegion(**options)) for lowered in pair]
    declared, placeholder = fits
    _close(declared.params, placeholder.params)
    _close(declared.cost, placeholder.cost)
    sds = [inf.laplace(lowered, fit).marginal_sd() for lowered, fit in zip(pair, fits)]
    _close(sds[0], sds[1])


class _StopAtSecondStep(inf.Callback):
    def __init__(self):
        self.steps = 0

    def on_step(self, info):
        self.steps += 1
        if self.steps == 2:
            raise inf.StopFit()


def test_a_fit_resumed_from_its_checkpoint_equals_the_uninterrupted_fit(pair, tmp_path):
    """A fit of partial reads stopped after two steps resumes from its checkpoint to the same end."""
    declared = pair[0]
    straight = small_fit(declared, inf.TrustRegion())
    stop = (_StopAtSecondStep(),)
    stopped = small_fit(declared, inf.TrustRegion(), callbacks=stop, checkpoint=tmp_path)
    assert stopped.status is inf.TerminationStatus.STOPPED
    resumed = small_fit(declared, inf.TrustRegion(), checkpoint=tmp_path)
    assert resumed.n_steps == straight.n_steps
    np.testing.assert_array_equal(resumed.params, straight.params)


def _study(models, placement, reads="declared"):
    """Three experiments of ``models`` (one per experiment), ``source`` placed as named."""
    cells = ("c1", "c2", "c2")
    experiments = tuple(
        decay_experiment(model, f"e{index + 1}", parameters={"cell": cell})
        for index, (model, cell) in enumerate(zip(models, cells))
    )
    handles = decay_recipe("reads_used", source=True)
    rate, source = handles.rate, handles.source
    if placement == "parameter set":
        return inf.Study(
            experiments,
            parameters=ds.ParameterSet(shared=(rate, source)),
            independent=True,
            reads=reads,
            clusters=None,
        )
    shared = (rate,) if placement in ("cell", "experiment", "nowhere") else (rate, source)
    per = {placement: (source,)} if placement in ("cell", "experiment") else {}
    hierarchy = inf.Hierarchy(levels=("cell",), shared=shared, per=per)
    return inf.Study(experiments, parameters=hierarchy, reads=reads)


@pytest.mark.parametrize(
    ("models", "placement", "reads", "match"),
    [
        ("bbb", "shared", "declared", "'source' is placed in shared= and no experiment"),
        ("ubb", "cell", "declared", "no experiment of its member 'c2' declares it"),
        ("uub", "experiment", "declared", "experiment 'e3' does not declare it"),
        ("uub", "parameter set", "declared", "places the handles in an inf.Hierarchy"),
        ("uub", "nowhere", "declared", "'source' of experiment 'e1' is learnable"),
        ("uub", "shared", "all", r"'source' is absent from experiment 'e3'; declared names: \["),
        ("uuu", "shared", "some", "reads= is 'all' or 'declared'; got 'some'"),
    ],
    ids=("unread", "member", "experiment-level", "parameter-set", "unplaced", "all", "value"),
)
def test_a_study_refuses_a_handle_no_experiment_reads(models, placement, reads, match):
    """Under "declared" every placed block needs a reader and every learnable declaration a
    place; under "all" an absent handle keeps its message."""
    recipes = {
        "u": decay_recipe("reads_used", source=True),
        "b": decay_recipe("reads_bare", source=False),
    }
    with pytest.raises(ValueError, match=match):
        _study(tuple(recipes[key] for key in models), placement, reads)
