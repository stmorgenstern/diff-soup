"""Linked experiments in a world of three processes: the source and the lanes it drives on different ranks.

``mpirun -n 3`` launches a script that lowers the linked study of
``tests.support.links`` with ``ds.distributed.run``: the source on one process,
each driven cell on another. Process 0 checks the Gauss-Newton gradient against
central differences of the value cost and fits the study, joint and nested, from a
wrong start. The outputs of the source travel in its process's reply, and the link
values to the driven lanes' processes after the points of the second request. The
world runs twice: driven models with the source's rate as an inert placeholder, and
without it under ``reads="declared"``, where the link adds the rate to their lanes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.support.links import TRUE_POINT

_ROOT = Path(__file__).resolve().parents[2]

_WORLD = r"""
import json
import sys

import numpy as np

import diff_soup as ds
from diff_soup.optimization import inference as inf
from tests.support.links import START, linked_recipes, linked_study

point = np.asarray([0.7, 0.5, 1.1, 1.0])
step = 4e-3
placeholder = sys.argv[1] == "placeholder"
recipes = linked_recipes("linked_world", placeholder=placeholder)
study = linked_study(recipes, reads="all" if placeholder else "declared")
with ds.distributed.run(study) as lowered:
    report = lowered.prepare(modes=("value", "blocks", "gradient", "frozen"))
    _, gradient, _, failed = lowered.gauss_newton_fn(point)
    eye = np.eye(point.size)
    shifts = np.concatenate(
        [point + step * eye, point - step * eye, point + 2 * step * eye, point - 2 * step * eye]
    )
    values = lowered.value_and_gradient(shifts).values
    n = point.size
    plus, minus, plus2, minus2 = (values[k * n : (k + 1) * n] for k in range(4))
    fits = {}
    for name, nested_at in (("joint", None), ("nested", "cell")):
        fit = inf.optsolve(
            lowered.with_start(START),
            inf.TrustRegion(max_steps=40, nested_at=nested_at),
            diagnostics=False,
        )
        fits[name] = np.asarray(fit.params).tolist()
    record = {
        "processes": lowered.processes,
        "ranks": {str(lane): int(rank) for lane, rank in report.placement.rank_of.items()},
        "failed": int(failed),
        "gradient": np.asarray(gradient).tolist(),
        "differences": ((8.0 * (plus - minus) - (plus2 - minus2)) / (12.0 * step)).tolist(),
        "fits": fits,
    }
    print("REPORT " + json.dumps(record), flush=True)
"""


VARIANTS = ("placeholder", "declared")


def _world(directory: Path, variant: str) -> dict:
    """The report of process 0 of a three-process run of the linked study of ``variant``."""
    script = directory / "world.py"
    script.write_text(_WORLD)
    launcher = ("SLURM_", "OMPI_", "PMI", "PMIX_", "DIFFSOUP_")
    env = {k: v for k, v in os.environ.items() if not k.startswith(launcher)}
    env = {k: v for k, v in env.items() if "proxy" not in k.lower()}
    path = os.pathsep.join(item for item in (str(_ROOT), env.get("PYTHONPATH", "")) if item)
    ranks = {
        "PYTHONPATH": path,
        "JAX_PLATFORMS": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "JAX_ENABLE_X64": "1",
        "OMP_NUM_THREADS": "1",
        "DIFFSOUP_CACHE_DIR": str(directory / "cache"),
    }
    env.update(ranks, OMPI_MCA_btl_vader_single_copy_mechanism="none")
    command = ["mpirun", "-n", "3", "--oversubscribe", "env"]
    command += [f"{key}={value}" for key, value in ranks.items()]
    command += [sys.executable, "-m", "mpi4py", str(script), variant]
    run = subprocess.run(command, cwd=_ROOT, env=env, capture_output=True, text=True, timeout=140)
    assert run.returncode == 0, run.stderr[-4000:]
    line = next(item for item in run.stdout.splitlines() if item.startswith("REPORT "))
    return json.loads(line.removeprefix("REPORT "))


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    """The reports of the two worlds, by variant."""
    return {variant: _world(tmp_path_factory.mktemp(variant), variant) for variant in VARIANTS}


@pytest.mark.requires_mpi_launch
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_source_and_the_lanes_it_drives_run_on_different_processes(worlds, variant):
    """Three processes each hold one lane: the source and the two cells it drives."""
    assert worlds[variant]["processes"] == 3
    assert sorted(worlds[variant]["ranks"].values()) == [0, 1, 2]


@pytest.mark.requires_mpi_launch
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_gauss_newton_gradient_over_three_processes_matches_central_differences(
    worlds, variant
):
    """The level sums of the lanes' blocks carry the link's chain rule across processes: their
    gradient matches the five-point central difference of the value cost."""
    world = worlds[variant]
    assert world["failed"] == 0
    # measured 1.3e-7 relative (column b of c2), CPU 2026-11-20, mpirun -n 3
    np.testing.assert_allclose(world["gradient"], world["differences"], rtol=1e-6, atol=1e-6)


@pytest.mark.requires_mpi_launch
def test_the_declared_world_gradient_equals_the_placeholder_world_gradient(worlds):
    """Over three processes, the driven lanes that read the source's rate through the link alone
    give the gradient of the driven lanes that declare it."""
    # measured 3.2e-16 relative, CPU 2026-11-20, mpirun -n 3
    np.testing.assert_allclose(
        worlds["declared"]["gradient"], worlds["placeholder"]["gradient"], rtol=1e-12, atol=1e-9
    )


@pytest.mark.requires_mpi_launch
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", ["joint", "nested"])
def test_trust_region_fits_over_three_processes_recover_the_rates(worlds, variant, name):
    """The joint and the nested fits over three processes recover the true rates."""
    # measured 2.7e-9 relative (nested, b of c1), CPU 2026-11-20, mpirun -n 3
    np.testing.assert_allclose(worlds[variant]["fits"][name], TRUE_POINT, rtol=1e-6)
