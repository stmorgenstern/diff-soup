"""Optimization lanes over a `("data",)` mesh: the point and the gradient in `args` equal one
device's, on forced host devices."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]

#: Five HS35 lanes (padded to the mesh) on four forced host CPU devices, through `lanes` with a
#: `("data",)` layout and without one; the gradient of a sum over the lanes in `args` likewise.
_WORKER = r"""
import json
import jax
import jax.numpy as jnp
import diff_soup as ds
from tests.support.optimization import hs35

toy = hs35()
solver = ds.optimization.InteriorPoint(hessian="exact", tol=1e-12)
layout = ds.Layout(mesh=ds.global_mesh(("data",)))
batch = jnp.array([3.0, 2.5, 3.4, 3.6, 2.9])
x0 = jnp.broadcast_to(toy.problem.x0, (5, 3))
point = lambda b, layout: ds.optimization.lanes(solver, toy.problem, b, x0, layout=layout)[0]
loss = lambda b, layout: jnp.sum(point(b, layout) ** 2)
report = {
    "devices": jax.device_count(),
    "point": float(jnp.max(jnp.abs(point(batch, layout) - point(batch, None)))),
    "gradient": float(
        jnp.max(jnp.abs(jax.grad(loss)(batch, layout) - jax.grad(loss)(batch, None)))
    ),
}
print("REPORT " + json.dumps(report))
"""


@pytest.fixture(scope="module")
def four_devices():
    """The worker's report from one subprocess on four forced host devices."""
    env = {key: value for key, value in os.environ.items() if "proxy" not in key.lower()}
    env.update(
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=4",
        OMP_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="",
        JAX_ENABLE_X64="1",
    )
    run = subprocess.run(
        [sys.executable, "-c", _WORKER],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=280,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    line = next(item for item in run.stdout.splitlines() if item.startswith("REPORT "))
    return json.loads(line.removeprefix("REPORT "))


def test_lanes_over_a_data_mesh_equal_one_device(four_devices):
    """Each lane's point over four devices is the single-device point, bit for bit."""
    assert four_devices["devices"] == 4 and four_devices["point"] == 0.0


def test_the_gradient_through_a_data_mesh_equals_one_device(four_devices):
    """The gradient of a sum over the lanes in `args` is the single-device one, bit for bit."""
    assert four_devices["gradient"] == 0.0
