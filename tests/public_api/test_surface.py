"""Public surface: exported names, top-level verbs, import time, errors, hashable configuration."""

from __future__ import annotations

import importlib
import json
import os
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


def test_importing_jax_before_the_hoisting_flag_warns_once_and_names_the_fix():
    """A JAX imported without the hoisting flag draws one RuntimeWarning naming the setting."""
    late = subprocess.run(
        [sys.executable, "-W", "always", "-c", "import jax; import diff_soup"],
        env={**_CPU, "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS": "0"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert late.returncode == 0, late.stderr
    assert late.stderr.count("RuntimeWarning") == 1
    assert "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=1" in late.stderr


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
