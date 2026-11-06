"""Architecture: layer direction, static fields, content cache, adjoint identity, the facade."""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import diff_soup as ds
from diff_soup import timestepping
from diff_soup.optimization import inference as inf
from diff_soup.problem._cache import BoundedContentCache
from tests.support.inference import decay_experiment, decay_recipe

_PACKAGE_ROOT = Path(ds.__file__).parent
_LAYER_TOOL = _PACKAGE_ROOT.parent / ".github" / "scripts" / "check_layers.py"


@pytest.fixture(scope="module")
def layers():
    """The repository's layer tool, .github/scripts/check_layers.py, loaded from the checkout."""
    if not _LAYER_TOOL.is_file():
        pytest.skip(f"layer rows need a source checkout; {_LAYER_TOOL} does not exist")
    spec = importlib.util.spec_from_file_location("check_layers", _LAYER_TOOL)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_package_imports_follow_the_declared_layer_direction(layers):
    """Module-level imports run down LAYERS; the baseline's exceptions may only shrink."""
    assert layers.new_violations() == []


def test_the_fitting_engine_never_imports_the_statistics_layer(layers):
    """No module of inference.fitting or optimization.solvers but the facade imports statistics."""
    optimization = _PACKAGE_ROOT / "optimization"
    facade = optimization / "inference" / "fitting" / "problem.py"
    statistics = "diff_soup.optimization.inference.statistics"
    back_edges = []
    for subpackage in (optimization / "inference" / "fitting", optimization / "solvers"):
        for path in sorted(subpackage.rglob("*.py")):
            if path == facade:
                continue
            module, is_package = layers._module_name(path, _PACKAGE_ROOT)
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imported = layers._imported_modules(node, module, is_package, _PACKAGE_ROOT)
                    back_edges += [
                        f"{module} -> {name}" for name in imported if name.startswith(statistics)
                    ]
    assert back_edges == []


def test_array_annotations_are_never_declared_static():
    """No class field annotated Array or jax.Array is declared with static=True."""
    violations = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            for statement in node.body:
                if not (
                    isinstance(statement, ast.AnnAssign)
                    and ast.unparse(statement.annotation) in {"Array", "jax.Array"}
                    and isinstance(statement.value, ast.Call)
                ):
                    continue
                if any(
                    keyword.arg == "static" and getattr(keyword.value, "value", None) is True
                    for keyword in statement.value.keywords
                ):
                    violations.append(f"{path.relative_to(_PACKAGE_ROOT)}:{node.name}")
    assert violations == []


def test_the_content_cache_is_identity_stable_bounded_and_least_recently_used():
    """A hit returns the stored object; at capacity the least recently used entry goes."""
    cache = BoundedContentCache(capacity=2)
    first = cache.get_or_build("first", handle=0, build_fn=object)
    cache.get_or_build("second", handle=0, build_fn=object)
    assert cache.get_or_build("first", handle=1, build_fn=object) is first
    cache.get_or_build("third", handle=0, build_fn=object)
    assert ("first" in cache, "second" in cache, len(cache)) == (True, False, 2)
    assert (cache.hits(), cache.evictions()) == (1, 1)


@pytest.mark.parametrize("name", ["ForwardAdjoint", "CheckpointAdjoint"])
def test_every_public_adjoint_path_resolves_one_class(name):
    """ds and ds.timestepping export the same adjoint class object."""
    assert getattr(ds, name) is getattr(timestepping, name)


def test_the_public_inference_problem_is_a_facade_over_a_numeric_leaf():
    """InferenceProblem wraps a numeric problem that is not itself the public class."""
    decay = decay_recipe("architecture_facade")
    problem = inf.InferenceProblem(decay_experiment(decay), parameters=decay.rate)
    leaf = problem.numeric_problem
    assert leaf is not None and not isinstance(leaf, inf.InferenceProblem)
    assert callable(inf.Study.compile_inference_problem)


def test_every_problem_shares_one_array_free_base_and_the_module_problems_stay_pytrees():
    """InferenceProblem and the control problem subclass one abstract base that holds
    no state; a ControlProblem's leaves are its own arrays, so a batch of them is one
    filter_vmap and filter_jit traces it."""
    base = ds.optimization.AbstractOptimizationProblem
    control = ds.optimization.control
    assert issubclass(inf.InferenceProblem, base) and issubclass(inf.LoweredStudy, base)
    assert issubclass(control.ControlProblem, base)
    assert not issubclass(base, eqx.Module) and "__init__" not in vars(base)
    batch = control.ControlProblem(initial_state=jnp.ones((3, 2)), theta=jnp.arange(3.0))
    assert len(jax.tree.leaves(batch)) == 2
    doubled = eqx.filter_vmap(lambda problem: problem.args()[0] * problem.args()[1])(batch)
    assert jnp.array_equal(
        eqx.filter_jit(lambda value: value)(doubled), jnp.ones((3, 2)) * jnp.arange(3.0)[:, None]
    )
