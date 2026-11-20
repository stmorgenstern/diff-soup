"""Fit the study of partial reads in a run of several processes; process 0 prints the numbers.

Launched by ``tests/multi_device/test_study_reads_world.py`` as
``mpirun -n 3 python -m mpi4py tests/support/study_reads_world.py <placement>``. Every
process lowers ``reads_study("declared", placement)`` through ``ds.distributed.run``; process
0 evaluates the Gauss-Newton system at one point, fits joint and nested at ``cell``, takes
the Laplace standard deviations and prints them as one JSON line after ``READS``.
"""

from __future__ import annotations

import json
import sys

import numpy as np

import diff_soup as ds
from diff_soup.optimization import inference as inf
from tests.support.inference import reads_study, small_fit


def record(lowered) -> dict:
    """The numbers the one-process run is compared with."""
    point = np.asarray(lowered.x0) + np.linspace(0.01, 0.02, np.size(lowered.x0))
    cost, gradient, matrix, failed = lowered.gauss_newton_fn(point)
    joint = small_fit(lowered, inf.TrustRegion())
    nested = small_fit(lowered, inf.TrustRegion(nested_at="cell"))
    sd = inf.laplace(lowered, joint).marginal_sd()
    return {
        "processes": int(lowered.processes),
        "widths": {lane: b.width for b in lowered.plan().buckets for lane in b.lanes},
        "point": point.tolist(),
        "cost": float(cost),
        "gradient": np.asarray(gradient).tolist(),
        "matrix": np.asarray(matrix).tolist(),
        "failed": int(failed),
        "joint": {"params": np.asarray(joint.params).tolist(), "cost": float(joint.cost)},
        "nested": {"params": np.asarray(nested.params).tolist(), "cost": float(nested.cost)},
        "sd": np.asarray(sd).tolist(),
    }


if __name__ == "__main__":
    with ds.distributed.run(reads_study("declared", sys.argv[1]), objective=inf.MAP()) as study:
        print("READS " + json.dumps(record(study)), flush=True)
