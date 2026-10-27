"""Suite-wide policy, tier, capability and JAX configuration (the contract is in README.md)."""

from __future__ import annotations

import gc
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

# Before any import of jax: the library's programs hoist their mesh arrays out
# of the program text only when JAX sees this setting at import.
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")

import equinox as eqx  # noqa: E402
import jax
import pytest

TEST_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = TEST_ROOT / "suite_manifest.json"

jax.config.update("jax_enable_x64", True)

if cache_dir := os.getenv("JAX_COMPILATION_CACHE_DIR"):
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    minimum = float(os.getenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", minimum)

#: Tier directory -> (marker, per-test timeout in seconds); see README.md.
TIERS = {
    "public_api": ("unit", 120),
    "contracts": ("unit", 120),
    "performance_contracts": ("unit", 120),
    "multi_device": ("unit", 300),
    "integration": ("integration", 180),
    "validation": ("validation", 300),
}


@pytest.fixture(scope="module", autouse=True)  # per module: bounds retained executables
def _bound_transform_cache_lifetime():
    yield
    eqx.clear_caches()
    jax.clear_caches()
    gc.collect()


#: Parameter ids of the linear backends that run on the host CPU only.
_HOST_BACKEND_IDS = frozenset({"klu", "superlu"})


@pytest.fixture(autouse=True)
def _host_backends_on_the_cpu_device(request):
    """Run a test of a host-only linear backend on the CPU device.

    KLU and SuperLU_MT factor on the host CPU and refuse a program compiled
    for a GPU. On a host whose default device is a GPU, a test marked
    ``requires_klu`` or ``requires_superlu`` or parametrized with a ``klu`` /
    ``superlu`` backend id runs under ``jax.default_device`` of the first CPU
    device, so every module runs on both platforms. Such a test fails in
    setup, naming ``JAX_PLATFORMS``, when the run has no CPU device (JAX
    initializes only the platforms ``JAX_PLATFORMS`` lists).
    """
    callspec = getattr(request.node, "callspec", None)
    ids = set(callspec.id.split("-")) if callspec is not None else set()
    host = bool(ids & _HOST_BACKEND_IDS) or any(
        request.node.get_closest_marker(name) is not None
        for name in ("requires_klu", "requires_superlu")
    )
    if not host or jax.devices()[0].platform == "cpu":
        yield
        return
    try:
        cpu = jax.devices("cpu")[0]
    except (RuntimeError, IndexError):
        platforms = os.environ.get("JAX_PLATFORMS", "")
        pytest.fail(
            "host-only linear backend: this test runs KLU or SuperLU_MT on the CPU device, and "
            f"this run has no CPU platform (JAX_PLATFORMS={platforms!r}); list cpu in "
            "JAX_PLATFORMS (JAX_PLATFORMS=cuda,cpu on a GPU host) or leave it unset",
            pytrace=False,
        )
    with jax.default_device(cpu):
        yield


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _load_policy_module():
    spec = importlib.util.spec_from_file_location(
        "diff_soup_test_suite_policy_v2", TEST_ROOT / "suite_policy.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load tests/suite_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cuda_available() -> bool:
    try:
        return bool(jax.devices("cuda"))
    except Exception:
        return False


def _cudss_available() -> bool:
    if not _cuda_available():
        return False
    try:
        from diff_soup.linear import CuDSS

        return CuDSS is not None
    except Exception:
        return False


def _klu_available() -> bool:
    try:
        from diff_soup.linear.klu_backend import klu_available

        return bool(klu_available())
    except Exception:
        return False


def _superlu_available() -> bool:
    try:
        from diff_soup.linear.superlu_backend import superlu_available

        return bool(superlu_available())
    except Exception:
        return False


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the opt-in tiers and the two capability modes."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run optional-backend integration contracts",
    )
    parser.addoption(
        "--run-validation",
        action="store_true",
        default=False,
        help="run analytical physics validation",
    )
    parser.addoption(
        "--strict-capabilities",
        action="store_true",
        default=False,
        help="fail any skip in a selected tier that is not a capability-marker skip",
    )
    parser.addoption(
        "--require-capabilities",
        action="store_true",
        default=False,
        help=(
            "fail, rather than skip, a test whose build capability (requires_cuda, "
            "requires_cudss, requires_klu, requires_superlu) this host does not provide; a "
            "missing requires_two_devices, requires_three_devices, requires_four_devices or "
            "requires_two_processes stays an owed skip, since a device count or a process world "
            "belongs to the launch, not to the build"
        ),
    )


@pytest.fixture(scope="session", autouse=True)
def _library_cache_directory(tmp_path_factory: pytest.TempPathFactory):
    """Point the library's persisted records (window initializations) at a fresh directory.

    The suite's records never reach the user's cache, and every run starts from an empty
    store, also on a host that exports ``DIFFSOUP_CACHE_DIR``; the variable's previous
    value (or its absence) is restored after the session.
    """
    previous = os.environ.get("DIFFSOUP_CACHE_DIR")
    os.environ["DIFFSOUP_CACHE_DIR"] = str(tmp_path_factory.mktemp("diff_soup_cache"))
    yield
    if previous is None:
        os.environ.pop("DIFFSOUP_CACHE_DIR", None)
    else:
        os.environ["DIFFSOUP_CACHE_DIR"] = previous


def pytest_configure(config: pytest.Config) -> None:
    """Declare the suite's own markers and install optional runtime type checking.

    ``unit``, ``integration``, ``validation`` and the ``requires_cuda`` /
    ``requires_cudss`` / ``requires_klu`` / ``requires_superlu`` markers are
    declared in pyproject.toml; the device- and process-count markers are this
    suite's.
    """
    config.addinivalue_line(
        "markers", "requires_two_devices: needs at least two visible JAX devices"
    )
    config.addinivalue_line(
        "markers", "requires_three_devices: needs at least three visible JAX devices"
    )
    config.addinivalue_line(
        "markers", "requires_four_devices: needs at least four visible JAX devices"
    )
    config.addinivalue_line(
        "markers", "requires_two_processes: needs a JAX world of at least two processes"
    )
    if _has("beartype"):
        import jaxtyping

        jaxtyping.install_import_hook("diff_soup", "beartype.beartype")


def pytest_sessionstart(session: pytest.Session) -> None:  # noqa: ARG001
    """Reject suite inflation before collection or compilation begins."""
    _load_policy_module().validate(allow_missing=True)


_MISSING_CAPABILITY = pytest.StashKey[str]()
#: Capabilities a launch provides rather than a build: --require-capabilities leaves a
#: test marked with one of them, on a run that lacks it, an owed skip.
_LAUNCH_CAPABILITIES = frozenset(
    {
        "requires_two_devices",
        "requires_three_devices",
        "requires_four_devices",
        "requires_two_processes",
    }
)
_TIER_SELECTED = pytest.StashKey[bool]()
_MARKER_SKIP = pytest.StashKey[bool]()


def _tier(path: Path) -> tuple[str, Path]:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(TEST_ROOT):
        return "", resolved  # outside tests/: no tier, reported as a usage error
    relative = resolved.relative_to(TEST_ROOT)
    return relative.parts[0], relative


def _tier_selected(config: pytest.Config, tier: str) -> bool:
    if tier == "integration":
        return config.getoption("--run-integration")
    return tier != "validation" or config.getoption("--run-validation")


def pytest_collection_modifyitems(
    session: pytest.Session,  # noqa: ARG001
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Infer tiers from directories and gate tests on the capabilities they declare."""
    require = config.getoption("--require-capabilities")
    capabilities = {
        "requires_cuda": (_cuda_available(), "no CUDA-capable JAX device"),
        "requires_cudss": (_cudss_available(), "native cuDSS backend is unavailable"),
        "requires_klu": (
            _klu_available(),
            "needs the KLU extension (_native_klu) and its SuiteSparse libraries",
        ),
        "requires_superlu": (
            _superlu_available(),
            "needs the SuperLU_MT extension (_native_superlu) and its SuperLU_MT library",
        ),
        "requires_two_devices": (
            jax.device_count() >= 2,
            f"needs two visible devices, this run has {jax.device_count()}",
        ),
        "requires_three_devices": (
            jax.device_count() >= 3,
            f"needs three visible devices, this run has {jax.device_count()}",
        ),
        "requires_four_devices": (
            jax.device_count() >= 4,
            f"needs four visible devices, this run has {jax.device_count()}",
        ),
        "requires_two_processes": (
            jax.process_count() >= 2,
            f"needs a two-process world, this run has {jax.process_count()} process(es)",
        ),
    }

    for item in items:
        tier, relative = _tier(item.path)
        if tier not in TIERS:
            raise pytest.UsageError(f"test module is outside a declared tier: {relative}")
        marker, timeout = TIERS[tier]
        item.add_marker(getattr(pytest.mark, marker))
        item.stash[_TIER_SELECTED] = _tier_selected(config, tier)
        if not item.stash[_TIER_SELECTED]:
            item.add_marker(pytest.mark.skip(reason=f"pass --run-{tier} to enable"))
        if _has("pytest_timeout"):
            item.add_marker(pytest.mark.timeout(timeout))
        missing = [
            (name, reason)
            for name, (present, reason) in capabilities.items()
            if name in item.keywords and not present
        ]
        unbuilt = [(name, reason) for name, reason in missing if name not in _LAUNCH_CAPABILITIES]
        if require and unbuilt:
            item.stash[_MISSING_CAPABILITY] = "; ".join(
                f"required capabilities: test is marked {name} but this host does not provide "
                f"it ({reason})"
                for name, reason in unbuilt
            )
        elif missing:
            name, reason = missing[0]
            item.stash[_MARKER_SKIP] = True
            item.add_marker(pytest.mark.skip(reason=f"owed: {name} ({reason})"))


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Under --require-capabilities, fail a selected test whose build capability is missing."""
    if message := item.stash.get(_MISSING_CAPABILITY, None):
        pytest.fail(message, pytrace=False)


def _strict_skip_failure(config: pytest.Config, tier: str, report, marker_skip: bool) -> None:
    if not config.getoption("--strict-capabilities") or not _tier_selected(config, tier):
        return
    if marker_skip or not report.skipped or hasattr(report, "wasxfail"):
        return
    longrepr = report.longrepr
    reason = longrepr[2] if isinstance(longrepr, tuple) else longrepr
    report.outcome = "failed"
    report.longrepr = f"strict capabilities: unexpected skip in tier {tier!r}: {reason}"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):  # noqa: ARG001
    """Under --strict-capabilities, turn a non-marker skip in a selected tier into a failure."""
    report = yield
    _strict_skip_failure(
        item.config, _tier(item.path)[0], report, item.stash.get(_MARKER_SKIP, False)
    )
    return report


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector: pytest.Collector):
    """Under --strict-capabilities, turn a module-level skip in a selected tier into an error."""
    report = yield
    if isinstance(collector, pytest.Module):
        _strict_skip_failure(collector.config, _tier(collector.path)[0], report, False)
    return report


def pytest_collection_finish(session: pytest.Session) -> None:
    """Enforce the parameter-expanded collection budgets on full-suite runs."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = {str((TEST_ROOT.parent / path).resolve()) for path in manifest["test_files"]}
    collected_files = {str(Path(item.path).resolve()) for item in session.items}
    if collected_files != expected:
        return
    limit = manifest["budgets"]["collected_tests"]
    if len(session.items) > limit:
        raise pytest.UsageError(f"collected-test budget exceeded: {len(session.items)} > {limit}")
    counts = Counter(_tier(item.path)[0] for item in session.items)
    for tier, ceiling in sorted(manifest["tier_budgets"].items()):
        if counts[tier] > ceiling["collected"]:
            raise pytest.UsageError(
                f"{tier} tier collected-test ceiling exceeded: {counts[tier]} > "
                f"{ceiling['collected']}"
            )
