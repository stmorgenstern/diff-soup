"""A study whose experiments read the handles they declare, fitted in a world of three processes.

``mpirun`` launches ``tests/support/study_reads_world.py`` on three processes, one experiment
each; process 0 prints the Gauss-Newton system at one point, the joint and nested fits and
the Laplace standard deviations. Each equals that of the twin study with an inert
placeholder, lowered and fitted here in one process.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import diff_soup as ds
from diff_soup.optimization import inference as inf
from tests.support.inference import reads_study, small_fit

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "tests" / "support" / "study_reads_world.py"
#: measured 2.7e-16 relative on the system, 6.8e-13 on the fitted values, 6.5e-13 on the
#: cost, 3.9e-12 on the Laplace sd (shared placement), CPU workstation 2026-11-20, Open MPI,
#: pytest tests/multi_device/test_study_reads_world.py
AGREEMENT = 1e-10


def _close(actual, expected):
    """``actual`` equals ``expected`` to ``AGREEMENT`` relative to the largest entry."""
    expected = np.asarray(expected)
    scale = float(np.max(np.abs(expected))) or 1.0
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=AGREEMENT * scale)


def _launch(placement: str, cache: Path) -> dict:
    """Run the script under ``mpirun -n 3`` and return process 0's record."""
    command = ["mpirun", "-n", "3"]
    version = subprocess.run(["mpirun", "--version"], capture_output=True, text=True).stdout
    if "Open MPI" in version or "OpenRTE" in version:
        command.append("--oversubscribe")
    path = os.pathsep.join(p for p in (str(_ROOT), os.environ.get("PYTHONPATH", "")) if p)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OMPI_COMM_", "PMI", "SLURM_"))}
    env.update(
        PYTHONPATH=path, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1", DIFFSOUP_CACHE_DIR=str(cache)
    )
    # Shared-memory transfers between the ranks of one host fail in some containers.
    env.setdefault("OMPI_MCA_btl_vader_single_copy_mechanism", "none")
    command += [sys.executable, "-m", "mpi4py", str(_SCRIPT), placement]
    done = subprocess.run(command, cwd=_ROOT, env=env, capture_output=True, text=True, timeout=280)
    assert done.returncode == 0, done.stderr[-3000:]
    lines = [line for line in done.stdout.splitlines() if line.startswith("READS ")]
    assert len(lines) == 1, done.stdout[-2000:] + done.stderr[-2000:]
    return json.loads(lines[0][len("READS ") :])


@pytest.mark.requires_mpi_launch
@pytest.mark.parametrize("placement", ["shared", "level"])
def test_three_processes_fit_the_study_as_one_process_fits_its_placeholder_twin(
    placement, tmp_path
):
    """The system, the fits joint and nested at ``cell``, and the sd agree with one process."""
    assert shutil.which("mpirun") is not None
    world = _launch(placement, tmp_path)
    assert world["processes"] == 3 and world["widths"] == {"e1": 2, "e2": 2, "e3": 1}
    twin = ds.lower(reads_study("all", placement), objective=inf.MAP())
    cost, gradient, matrix, failed = twin.gauss_newton_fn(np.asarray(world["point"]))
    assert world["failed"] == failed == 0
    _close(world["cost"], cost)
    _close(world["gradient"], gradient)
    _close(world["matrix"], matrix)
    joint = small_fit(twin, inf.TrustRegion())
    nested = small_fit(twin, inf.TrustRegion(nested_at="cell"))
    for name, fit in (("joint", joint), ("nested", nested)):
        _close(world[name]["params"], fit.params)
        _close(world[name]["cost"], fit.cost)
    _close(world["sd"], inf.laplace(twin, joint).marginal_sd())
