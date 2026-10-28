"""Public surface: exported names, top-level verbs, import time, errors, hashable configuration."""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import diff_soup as ds

_MANIFEST = Path(__file__).resolve().parents[1] / "suite_manifest.json"
_NAMESPACES = json.loads(_MANIFEST.read_text(encoding="utf-8"))["public_namespaces"]
_CPU = {**os.environ, "JAX_PLATFORMS": "cpu"}

# measured 1.18, 1.21 and 1.33 s for a fresh `import diff_soup` (three runs), CPU VM 2026-10-22,
# while a second test stream ran; the bound is about four times the slowest
_IMPORT_SECONDS = 5.0


@pytest.mark.parametrize("name", _NAMESPACES)
def test_every_public_namespace_exports_names_it_holds(name):
    """A public namespace declares __all__, holds every name in it and exports no private name."""
    module = importlib.import_module(name)
    exported = set(module.__all__)
    assert exported and not {item for item in exported if item.startswith("_")}
    assert {item for item in exported if not hasattr(module, item)} == set()


def test_top_level_verbs_are_callable_and_errors_share_one_catchable_base():
    """ds.problem, ds.pdesolve, ds.lower and engine stock() are callable; every public error
    derives from ds.DiffSoupError."""
    assert isinstance(ds.problem("surface_contract"), ds.ProblemSpec)
    assert callable(ds.pdesolve) and callable(ds.lower)
    assert all(callable(engine.stock) for engine in (ds.JNDF, ds.JBDF, ds.JIDA))
    errors = (
        ds.ConfigurationError,
        ds.LoweringError,
        ds.BackendError,
        ds.OptionalDependencyError,
        ds.AlgebraicProjectionError,
    )
    assert all(issubclass(error, ds.DiffSoupError) for error in errors)


def test_a_fresh_import_is_fast_and_silent_with_the_hoisting_flag_set():
    """`import diff_soup` with the flag set raises no warning and takes under the bound."""
    script = (
        "import time; start = time.perf_counter(); import diff_soup; "
        "print(time.perf_counter() - start)"
    )
    run = subprocess.run(
        [sys.executable, "-W", "error", "-c", script],
        env={**_CPU, "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr
    assert float(run.stdout.strip().splitlines()[-1]) < _IMPORT_SECONDS


# measured 387 bytes for the closure below after `import jax; import diff_soup` (fix branch) and
# 80,403 bytes on 1.0.0, which embeds the array, CPU workstation 2026-10-28
_HOISTED_PROGRAM_BYTES = 4096


def test_importing_jax_first_still_hoists_closed_over_arrays_without_a_warning():
    """With jax imported before diff_soup and the hoisting flag off, `import diff_soup` is
    silent and a jitted closure over a 10,000-element array lowers to a small program."""
    script = (
        "import jax; import diff_soup; import jax.numpy as jnp; "
        "c = jnp.linspace(0.0, 1.0, 10_000); "
        "print(len(jax.jit(lambda y: y + jnp.sin(c)).lower(jnp.ones(10_000)).as_text()))"
    )
    late = subprocess.run(
        [sys.executable, "-W", "error", "-c", script],
        env={**_CPU, "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS": "0"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert late.returncode == 0, late.stderr
    assert int(late.stdout.strip().splitlines()[-1]) < _HOISTED_PROGRAM_BYTES


def _warning_sites(tmp_path, body, pattern):
    """File names of the warnings matching ``pattern`` that a script with ``body`` raises."""
    script = tmp_path / "user_script.py"
    script.write_text(body, encoding="utf-8")
    run = subprocess.run(
        [sys.executable, "-W", "always", str(script)],
        env={**_CPU, "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert run.returncode == 0, run.stderr
    lines = re.findall(r"^(.+?):\d+: \w+: (.*)$", run.stderr, flags=re.MULTILINE)
    return script, [path for path, message in lines if re.search(pattern, message)]


def test_a_float32_cpu_solve_warns_at_the_users_own_line(tmp_path):
    """The two float32 warnings of a solve name the user's script, not a diff_soup, equinox or
    jax file, and a float32 grid solve leaks no float64 request."""
    body = (
        "import jax.numpy as jnp\n"
        "import numpy as np\n"
        "import diff_soup as ds\n"
        "problem = ds.MassProblem(\n"
        "    rhs=lambda t, y, theta, control: -theta * y,\n"
        "    mass=ds.MassOperator.identity(2),\n"
        "    structure=ds.DAEStructure(differential_vars=np.ones((2,), dtype=bool)),\n"
        "    precision=ds.Precision.fp32(),\n"
        ")\n"
        "ds.pdesolve(problem, ds.JNDF(linear='dense'), (0.0, 0.5),\n"
        "            jnp.ones((2,), jnp.float32), theta=0.3)\n"
    )
    script, sites = _warning_sites(tmp_path, body, r"jax_enable_x64=False|a float32 problem")
    assert len(sites) == 2
    assert set(sites) == {str(script)}

    # With x64 off a float32 grid solve raises no JAX "Explicitly requested dtype float64" warning
    # from inside diff_soup.
    grid = (
        "import jax.numpy as jnp\n"
        "import numpy as np\n"
        "import diff_soup as ds\n"
        "spec = ds.problem('heat').domain('line', ds.cartesian_grid_1d(16, 0.0, 2.0 * np.pi, method='fvm'))\n"
        "spec.field('T', mass='differential', scale=1.0, init=lambda x: 0.3 + jnp.sin(x))\n"
        "spec.equation(writes='T', reads=('T',))(lambda q: 0.5 * q.ops.laplacian(q['T']))\n"
        "spec.bc.periodic('T')\n"
        "ds.pdesolve(spec.build(precision=ds.Precision.fp32_ir()), ds.JNDF.stock(linear='dense'),\n"
        "            (0.0, 0.4), spec.assemble_state({}), saveat=ds.SaveAt(t1=True))\n"
    )
    _, leaks = _warning_sites(tmp_path, grid, "Explicitly requested dtype")
    assert leaks == []


def test_an_omitted_field_scale_warns_at_the_users_own_line(tmp_path):
    """The warning for a field declared without ``scale=`` names the line in the user's
    script that builds the recipe."""
    body = (
        "import diff_soup as ds\n"
        "spec = ds.problem('unscaled').domain('line', ds.cartesian_grid_1d(4, 0.0, 1.0, method='fvm'))\n"
        "spec.field('u', mass='differential')\n"
        "spec.equation(writes='u', reads=('u',))(lambda q: -q.u)\n"
        "spec.build()\n"
    )
    script, sites = _warning_sites(tmp_path, body, "have no scale=")
    assert sites == [str(script)]


def test_static_public_configuration_is_hashable_and_equal_to_itself():
    """Layouts, precisions, stock solvers and output policies hash and compare by value."""
    for build in (
        ds.Layout,
        ds.Precision,
        ds.Precision.fp32,
        lambda: ds.JNDF.stock(linear="dense"),
        lambda: ds.JIDA.stock(linear="dense"),
        lambda: ds.SaveAt(output=ds.DenseOutputPolicy(mode="projected")),
    ):
        first, second = build(), build()
        assert first == second and hash(first) == hash(second)
