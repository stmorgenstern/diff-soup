"""parallel.py — map identifiability workloads onto batched and parallel execution.

The sensitivity/Fisher pipeline is embarrassingly parallel at several nested
levels, and each level composes with the one below it:

* **Directions (automatic).** ``jax.jacfwd`` of a sensitivity run batches the
  per-step linear solves over the sensitivity directions (one factorization,
  many right-hand sides). This costs nothing to use and is always on.
* **Samples (one device).** ``jax.vmap`` of a per-sample criterion over prior
  samples or protocol candidates —
  [batched_criterion][diff_soup.inference.design.parallel.batched_criterion] /
  [robust_criterion][diff_soup.inference.design.parallel.robust_criterion].
* **Devices.** ``jax.shard_map`` of the sample axis over the visible devices
  with a ``psum`` mean —
  [sharded_robust_gradient][diff_soup.inference.design.parallel.sharded_robust_gradient].
  The global state is never gathered.
* **Processes.** Independent runs (multi-start design, profile likelihoods,
  heterogeneous models) on disjoint submeshes through
  [checkpointed_map][diff_soup.inference.design.parallel.checkpointed_map]. No
  JAX-transformation coupling at all, so it is also the fallback when an outer
  transformation does not compose.
* **Campaigns.** Long collections of independent fits use
  [checkpointed_map][diff_soup.inference.design.parallel.checkpointed_map]:
  every accepted or failed task is atomically recorded under a stable task ID,
  accepted tasks survive restarts, and process workers are confined to
  disjoint device groups. Aggregate results use
  [deterministic_weighted_reduce][diff_soup.inference.design.parallel.deterministic_weighted_reduce],
  whose summation order is fixed and independent of task completion order.

Choosing a pattern:

============================================  =============================
Workload                                      Pattern
============================================  =============================
Sensitivities / FIM at one parameter point    directions (automatic)
Robust design over prior samples              samples; + devices on several
Criterion sweep over protocol candidates      samples (+ devices)
Multi-start design, independent convergence   processes
Profile likelihoods / many independent fits   processes
============================================  =============================

[choose_pattern][diff_soup.inference.design.parallel.choose_pattern] encodes
the same table programmatically.
"""

from __future__ import annotations

import contextvars
import dataclasses
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._distributed import (
    _decision_scope,
    global_mesh,
    is_primary,
    local_value,
    mesh_groups,
    owns,
)
from diff_soup._logging import log

__all__ = [
    "batched_criterion",
    "robust_criterion",
    "sharded_robust_gradient",
    "choose_pattern",
    "CampaignFailure",
    "CampaignTaskRecord",
    "CampaignResult",
    "checkpointed_map",
    "deterministic_weighted_reduce",
]

_CAMPAIGN_SCHEMA_VERSION = 1
_campaign_submesh = contextvars.ContextVar("diff_soup_campaign_submesh", default=None)

_PATTERNS = {
    "point": "directions",
    "fim": "directions",
    "sensitivities": "directions",
    "robust": "samples",
    "prior": "samples",
    "candidates": "samples",
    "sweep": "samples",
    "multistart": "processes",
    "multi-start": "processes",
    "profile": "processes",
    "fits": "processes",
}


@dataclasses.dataclass(frozen=True)
class CampaignFailure:
    """Durable description of one failed campaign task.

    Attributes:
        task_id: Stable identity supplied by the task declaration.
        error_type: Exception class name raised by the worker.
        message: Rendered exception message.
        attempt: One-based attempt number recorded in the ledger.
        device_group: Global device IDs in the assigned submesh, or ``None``
            for a serial task.
    """

    task_id: str
    error_type: str
    message: str
    attempt: int
    device_group: Optional[tuple] = None


@dataclasses.dataclass(frozen=True)
class CampaignTaskRecord:
    """Typed view of one accepted or failed checkpoint-ledger entry."""

    task_id: str
    status: str
    attempt: int
    input_fingerprint: str
    device_group: Optional[tuple]
    result: Any = None
    failure: Optional[CampaignFailure] = None


@dataclasses.dataclass(frozen=True)
class CampaignResult:
    """Ordered accepted outputs plus the complete requested-task ledger view.

    ``results`` follows the order of ``items`` passed to
    [checkpointed_map][diff_soup.inference.design.parallel.checkpointed_map],
    never process completion order. Failed tasks do
    not acquire placeholder scientific values; they remain explicit under
    ``failures`` and ``records``.
    """

    results: tuple
    records: tuple[CampaignTaskRecord, ...]
    failures: tuple[CampaignFailure, ...]
    ledger_path: str

    @property
    def ok(self) -> bool:
        """Whether every requested task was accepted."""
        return not self.failures

    def __iter__(self):
        """Iterate over accepted outputs in input order."""
        return iter(self.results)

    def __len__(self) -> int:
        """Number of accepted outputs."""
        return len(self.results)

    def __getitem__(self, index):
        """Index accepted outputs in input order."""
        return self.results[index]


def batched_criterion(criterion: Callable, batch: Any, *args: Any) -> Any:
    """Evaluate a per-sample function over a leading sample axis with ``vmap``.

    Args:
        criterion: ``criterion(sample, *args) -> value`` — any pytree-valued
            per-sample computation (a scalar criterion, a FIM, channel
            sensitivities).
        batch: Pytree of stacked samples (leading axis = sample).
        *args: Broadcast (unbatched) extra arguments, e.g. the design vector.

    Returns:
        The stacked per-sample values (leading axis = sample).
    """
    in_axes = (0,) + (None,) * len(args)
    return jax.vmap(criterion, in_axes=in_axes)(batch, *args)


def robust_criterion(criterion: Callable, batch: Any, *args: Any) -> jax.Array:
    """Mean of a scalar per-sample criterion over the sample axis.

    The robust (average-case) design objective: differentiate this with
    ``jax.grad`` with respect to the design arguments.

    Args:
        criterion: ``criterion(sample, *args) -> scalar``.
        batch: Pytree of stacked samples (leading axis = sample).
        *args: Broadcast extra arguments (e.g. the design vector).

    Returns:
        The scalar mean criterion.
    """
    values = batched_criterion(criterion, batch, *args)
    if values.ndim != 1:
        raise ValueError("A robust criterion must return one scalar per sample.")
    if not values.shape[0]:
        raise ValueError("A robust criterion needs at least one sample.")
    return jnp.mean(values)


def _native_factor_residencies(jaxpr: Any) -> set:
    """The factor residencies other than ``"arrays"`` of the linear backends ``jaxpr`` solves with.

    Reads the linear-solver calls of ``jaxpr`` and of the jaxprs its
    equations hold (loop bodies, branches, nested and differentiated calls):
    each names its backend through its ``analysis`` parameter.
    """
    found: set = set()
    pending, seen = [jaxpr], set()
    while pending:
        node = pending.pop()
        node = getattr(node, "jaxpr", node)
        if id(node) in seen:
            continue
        seen.add(id(node))
        for eqn in getattr(node, "eqns", ()):
            backend = getattr(eqn.params.get("analysis"), "backend", None)
            residency = getattr(backend, "factor_residency", "arrays")
            if residency != "arrays":
                found.add(residency)
            for value in eqn.params.values():
                for item in value if isinstance(value, (tuple, list)) else (value,):
                    if hasattr(getattr(item, "jaxpr", item), "eqns"):
                        pending.append(item)
    return found


def sharded_robust_gradient(
    criterion: Callable,
    design: jax.Array,
    batch: Any,
    *,
    devices: Optional[Sequence] = None,
) -> tuple:
    """Value and design gradient of the mean criterion, sharded over devices.

    Shards the sample axis of ``batch`` across the devices, evaluates the
    per-sample criterion and its design derivative locally on each device,
    then reduces both means with ``psum``. The per-sample states never leave their device —
    only partial sums of the value and design gradient are reduced.

    The devices of one process run their shards in one execution, so a
    criterion that solves takes backends whose factors are arrays, such as
    ``Dense``, while ``devices`` holds two or more devices of this process: a
    backend that keeps its factors in native storage (``KLU``, ``SuperLU``,
    ``CuDSS``, or a backend over one of them) keeps them in storage the
    devices would share, as on a layout's data axis of several devices.

    Args:
        criterion: ``criterion(sample, design) -> scalar``.
        design: The design vector to differentiate with respect to.
        batch: Pytree of stacked samples; the leading axis must divide evenly
            by the device count.
        devices: Devices to shard over (default: all visible devices).

    Returns:
        ``(value, gradient)`` — the mean criterion and its design gradient.

    Raises:
        ValueError: If the sample axis does not divide by the device count.
        ConfigurationError: If ``devices`` holds two or more devices of this
            process and the criterion solves with a backend whose factors are
            not arrays.
    """
    from jax import shard_map
    from jax.sharding import AxisType, Mesh, PartitionSpec

    from diff_soup._jax_compat import jaxpr_as_fun

    design = jnp.asarray(design)
    devices = list(jax.devices()) if devices is None else list(devices)
    n_dev = len(devices)
    if not n_dev:
        raise ValueError("sharded_robust_gradient needs at least one device.")
    leaves = jax.tree_util.tree_leaves(batch)
    if not leaves:
        raise ValueError("sharded_robust_gradient: batch has no array leaves.")
    n_samples = int(leaves[0].shape[0]) if leaves[0].ndim else 0
    if not n_samples or any(not leaf.ndim or leaf.shape[0] != n_samples for leaf in leaves):
        raise ValueError("Every batch leaf must have the same nonempty leading sample axis.")
    if n_samples % n_dev != 0:
        raise ValueError(
            f"sharded_robust_gradient: {n_samples} samples do not shard evenly "
            f"over {n_dev} devices; pad or trim the batch to a multiple."
        )

    mesh = Mesh(devices, ("sample",), axis_types=(AxisType.Auto,))

    def one(sample, point):
        value = criterion(sample, point)
        if jnp.ndim(value) != 0:
            raise ValueError("A robust criterion must return one scalar per sample.")
        return value

    # Trace the derivative before entering a manual mesh. Adaptive checkpoint
    # closures then see their original scalar bookkeeping types, while the
    # staged solver loops execute entirely within each sample shard.
    sample_shapes = jax.tree.map(
        lambda value: jax.ShapeDtypeStruct((n_samples // n_dev, *value.shape[1:]), value.dtype),
        batch,
    )
    point_shape = jax.ShapeDtypeStruct(design.shape, design.dtype)
    derivative = jax.make_jaxpr(jax.vmap(jax.value_and_grad(one, argnums=1), in_axes=(0, None)))(
        sample_shapes, point_shape
    )
    here = sum(device.process_index == jax.process_index() for device in devices)
    if here > 1:
        native = _native_factor_residencies(derivative)
        if native:
            from diff_soup._errors import ConfigurationError

            raise ConfigurationError(
                f"sharded_robust_gradient runs the criterion on {here} devices of this process "
                "in one execution, and the criterion solves with a backend that keeps its "
                f"factors in {' and '.join(sorted(native))} storage, which those devices would "
                "share; solve with Dense, whose factors are arrays, inside the criterion, or "
                "pass devices= holding one device of this process."
            )
    evaluate = jaxpr_as_fun(derivative)

    def local_statistics(samples, point):
        values, gradients = evaluate(*jax.tree.leaves((samples, point)))
        totals = (jnp.sum(values, axis=0), jnp.sum(gradients, axis=0))
        return jax.tree.map(lambda total: jax.lax.psum(total, "sample") / n_samples, totals)

    reduce = shard_map(
        local_statistics,
        mesh=mesh,
        in_specs=(jax.tree.map(lambda _: PartitionSpec("sample"), batch), PartitionSpec()),
        out_specs=(PartitionSpec(), PartitionSpec()),
        check_vma=False,
    )
    return jax.jit(reduce)(batch, design)


def _json_value(value: Any) -> Any:
    """Convert a campaign declaration/result to strict deterministic JSON data."""
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "checkpointed_map: campaign mappings need string keys for "
                    f"stable JSON; got key {key!r}."
                )
            converted[key] = _json_value(item)
        return converted
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, jax.Array):
        return _json_value(local_value(value))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(
                f"checkpointed_map: ledger values must be finite JSON numbers; got {value!r}."
            )
        return value
    raise TypeError(
        "checkpointed_map: tasks and results must be JSON-serializable; "
        f"cannot encode {type(value).__name__}."
    )


def _canonical_json(value: Any) -> str:
    """Return the strict canonical JSON representation used for fingerprints."""
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _input_fingerprint(item: Any) -> str:
    """Hash a task declaration so a reused task ID cannot change meaning."""
    return hashlib.sha256(_canonical_json(item).encode("utf-8")).hexdigest()


def _task_id(item: Any) -> str:
    """Extract the required stable string task ID from a declaration."""
    if isinstance(item, Mapping):
        task_id = item.get("task_id")
    else:
        task_id = getattr(item, "task_id", None)
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(
            f"checkpointed_map: every item needs a non-empty string task_id; got {task_id!r}."
        )
    return task_id


def _empty_ledger() -> dict:
    """Return a new deterministic campaign ledger document."""
    return {"schema_version": _CAMPAIGN_SCHEMA_VERSION, "tasks": {}}


def _read_ledger(path: Path) -> dict:
    """Read and validate a campaign ledger."""
    if not path.exists():
        return _empty_ledger()
    try:
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"checkpointed_map: cannot resume malformed ledger {path}: {exc}."
        ) from exc
    if not isinstance(document, dict):
        raise ValueError(f"checkpointed_map: ledger {path} must contain a JSON object.")
    if document.get("schema_version") != _CAMPAIGN_SCHEMA_VERSION:
        raise ValueError(
            f"checkpointed_map: ledger {path} uses schema version "
            f"{document.get('schema_version')!r}, expected {_CAMPAIGN_SCHEMA_VERSION}."
        )
    if not isinstance(document.get("tasks"), dict):
        raise ValueError(f"checkpointed_map: ledger {path} has no valid tasks mapping.")
    return document


def _replace_ledger(path: Path, document: dict) -> None:
    """Atomically replace a ledger on its owning process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                document,
                handle,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_ledger(path: Path, document: dict) -> None:
    """Write on process zero and propagate write failures to every rank."""
    error = None
    if is_primary():
        try:
            _replace_ledger(path, document)
        except OSError as exc:
            error = f"checkpointed_map cannot write ledger {path}: {exc}"
    error = _broadcast_record(error)
    if error is not None:
        raise OSError(error)


def _normalize_device_groups(device_groups):
    """Validate disjoint Auto submeshes of the initialized global device world."""
    if device_groups is None:
        return None
    from jax.sharding import AxisType, Mesh

    groups = tuple(device_groups)
    if not groups or any(not isinstance(group, Mesh) or not group.devices.size for group in groups):
        raise ValueError("checkpointed_map device_groups must be nonempty JAX submeshes.")
    available, assigned = set(jax.devices()), set()
    for group in groups:
        if not set(group.axis_names) <= {"data", "model"} or any(
            kind != AxisType.Auto for kind in group.axis_types
        ):
            raise ValueError("Campaign submeshes need Auto data/model axes.")
        devices = set(group.devices.flat)
        if not devices <= available or assigned & devices:
            raise ValueError(
                "Campaign submeshes must be pairwise disjoint parts of the global mesh."
            )
        assigned.update(devices)
    return groups


def current_campaign_mesh():
    """The submesh assigned to the current campaign task, or None outside a task."""
    return _campaign_submesh.get()


def _invoke_campaign_task(fn, item, group):
    """Run in a thread-local mesh context without changing GPU visibility."""
    if group is None:
        return fn(item)
    device = next(
        device for device in group.devices.flat if device.process_index == jax.process_index()
    )
    token = _campaign_submesh.set(group)
    try:
        with _decision_scope(group), jax.set_mesh(group), jax.default_device(device):
            return fn(item)
    finally:
        _campaign_submesh.reset(token)


def _broadcast_record(value, source=0):
    """Broadcast small JSON task metadata through the initialized JAX world."""
    if jax.process_count() == 1:
        return value
    from diff_soup._jax_compat import multihost_utils

    sender = jax.process_index() == source
    payload = _canonical_json(value).encode() if sender else b""
    size = multihost_utils.broadcast_one_to_all(
        np.asarray([len(payload)], np.int64), is_source=sender
    )
    buffer = np.frombuffer(payload, np.uint8).copy() if sender else np.zeros(int(size[0]), np.uint8)
    received = multihost_utils.broadcast_one_to_all(buffer, is_source=sender)
    return json.loads(np.asarray(received, dtype=np.uint8).tobytes().decode())


def _failure_source(failed):
    """Identify a failing task rank before broadcasting its durable record."""
    candidate = jax.process_index() if failed else jax.process_count()
    if jax.process_count() > 1:
        from diff_soup._jax_compat import multihost_utils

        candidate = int(
            np.min(multihost_utils.process_allgather(np.asarray([candidate], np.int32)))
        )
    return candidate


def _running_record(
    *,
    task_id: str,
    fingerprint: str,
    attempt: int,
    device_group: Optional[tuple[int, ...]],
) -> dict:
    """Build the durable pre-execution record for one task attempt."""
    return {
        "task_id": task_id,
        "input_fingerprint": fingerprint,
        "status": "running",
        "attempt": int(attempt),
        "device_group": None if device_group is None else list(device_group),
    }


def _accept_record(record: dict, result: Any) -> dict:
    """Return an accepted ledger record preserving attempt metadata."""
    return {
        **record,
        "status": "accepted",
        "result": _json_value(result),
    }


def _fail_record(record: dict, error: BaseException) -> dict:
    """Return a failed ledger record with a durable exception description."""
    return {
        **record,
        "status": "failed",
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def _campaign_record(record: dict) -> CampaignTaskRecord:
    """Convert one validated JSON ledger entry into its typed public view."""
    group_value = record.get("device_group")
    group = None if group_value is None else tuple(group_value)
    error = record.get("error")
    failure = None
    if record.get("status") == "failed":
        error = {} if not isinstance(error, dict) else error
        failure = CampaignFailure(
            task_id=record["task_id"],
            error_type=str(error.get("type", "UnknownError")),
            message=str(error.get("message", "worker failed without a message")),
            attempt=int(record["attempt"]),
            device_group=group,
        )
    return CampaignTaskRecord(
        task_id=record["task_id"],
        status=record["status"],
        attempt=int(record["attempt"]),
        input_fingerprint=record["input_fingerprint"],
        device_group=group,
        result=record.get("result"),
        failure=failure,
    )


def checkpointed_map(
    fn: Callable,
    items: Sequence[Any],
    *,
    ledger_path: Any,
    resume: bool = True,
    execution: str = "serial",
    device_groups: Any = None,
    on_conflict: str = "error",
) -> CampaignResult:
    """Map a function over independent tasks, recording each outcome in a JSON ledger so a rerun resumes.

    Each item carries a `task_id` that stays the same across restarts. A result is stored as JSON
    and comes back in that form. A task that raises is recorded as failed, and the others still
    run.

    Args:
        fn: The task, a function of one item returning its result.
        items: The tasks, mappings or objects with a unique non-empty string `task_id`.
        ledger_path: The JSON file the ledger is written to, whole, after every change.
        resume: Whether to read the ledger first, keep its accepted tasks, and run the failed and
            interrupted ones again.
        execution: `"serial"` to run the tasks one after another, or `"processes"` to run each on
            a device group, on the processes that own it.
        device_groups: Disjoint JAX submeshes for `"processes"`, or `None` to give each device
            its own group.
        on_conflict: `"error"` to refuse a ledger that recorded a `task_id` for a different item,
            or `"restart"` to discard that ledger.

    Returns:
        A `CampaignResult` with the accepted `results` in item order, the `failures`, and one
            entry of `records` per task.
    """
    if execution not in ("serial", "processes"):
        raise ValueError(
            f"checkpointed_map: execution must be 'serial' or 'processes'; got {execution!r}."
        )
    if on_conflict not in ("error", "restart"):
        raise ValueError(
            f"checkpointed_map: on_conflict must be 'error' or 'restart'; got {on_conflict!r}."
        )
    groups = _normalize_device_groups(device_groups)
    if execution == "serial" and groups is not None:
        raise ValueError(
            "checkpointed_map: device groups require execution='processes'; "
            "use process execution to distribute tasks over submeshes."
        )

    if execution == "processes" and groups is None:
        mesh = global_mesh(("data",))
        groups = mesh_groups(mesh, (1,) * int(mesh.shape["data"]))
    items = tuple(items)
    declarations = []
    seen_ids = set()
    for index, item in enumerate(items):
        task_id = _task_id(item)
        if task_id in seen_ids:
            raise ValueError(
                f"checkpointed_map: duplicate task_id {task_id!r}; task identity "
                "must be unique within a campaign call."
            )
        seen_ids.add(task_id)
        declarations.append((index, task_id, item, _input_fingerprint(item)))

    path = Path(ledger_path)
    initial = None
    if is_primary():
        try:
            initial = {"document": _read_ledger(path) if resume else _empty_ledger()}
        except (OSError, ValueError) as error:
            initial = {"error": str(error)}
    initial = _broadcast_record(initial)
    if "error" in initial:
        raise ValueError(initial["error"])
    document = initial["document"]
    # Every process scans the same broadcast document with the same
    # declarations, so a restart is decided identically on all of them.
    conflicts = tuple(
        task_id
        for _index, task_id, _item, fingerprint in declarations
        if (entry := document["tasks"].get(task_id)) is not None
        and entry.get("input_fingerprint") != fingerprint
    )
    if conflicts:
        more = "" if len(conflicts) == 1 else f" ({len(conflicts) - 1} further task IDs too)"
        if on_conflict == "error":
            raise ValueError(
                f"checkpointed_map: task_id {conflicts[0]!r}{more} is recorded in the "
                f"ledger {path} under a different input declaration than this call "
                "declares: the campaign these tasks belong to has changed (for a "
                "bootstrap ledger, the fit, weights or optimizer its replicates were "
                f"declared under). Delete or move {path}, pass a fresh ledger_path "
                "(for bootstrap, a fresh checkpoint), or pass on_conflict='restart' to "
                "discard the stale ledger and run every task again."
            )
        held = {"accepted": 0, "failed": 0, "running": 0}
        for stored in document["tasks"].values():
            if stored.get("status") in held:
                held[stored["status"]] += 1
        log.warning(
            "checkpointed_map: task_id %r%s is recorded in the ledger %s under a "
            "different input declaration than this call declares; on_conflict='restart' "
            "discards that ledger, dropping %d accepted, %d failed and %d running "
            "records, and every task runs again from attempt 1.",
            conflicts[0],
            more,
            path,
            held["accepted"],
            held["failed"],
            held["running"],
        )
        document = _empty_ledger()
    task_ledger = document["tasks"]
    pending = []
    for index, task_id, item, fingerprint in declarations:
        old = task_ledger.get(task_id)
        if old is not None and old.get("status") == "accepted":
            continue
        attempt = 1 if old is None else int(old.get("attempt", 0)) + 1
        group = None if groups is None else groups[index % len(groups)]
        pending.append((index, task_id, item, fingerprint, attempt, group))

    # Materialize an empty/new ledger before the first task so a campaign
    # interrupted immediately still has a valid resumable document.
    _write_ledger(path, document)

    executors = {}
    try:
        if execution == "processes":
            executors = {
                index: ThreadPoolExecutor(max_workers=1)
                for index, group in enumerate(groups)
                if owns(group)
            }
        width = len(groups) if execution == "processes" else 1
        rounds = {}
        for task in pending:
            rounds.setdefault(task[0] // width, []).append(task)
        for tasks in rounds.values():
            futures, outcomes = {}, {}
            for _index, task_id, _item, fingerprint, attempt, group in tasks:
                identity = (
                    None
                    if group is None
                    else tuple(int(device.id) for device in group.devices.flat)
                )
                task_ledger[task_id] = _running_record(
                    task_id=task_id,
                    fingerprint=fingerprint,
                    attempt=attempt,
                    device_group=identity,
                )
            _write_ledger(path, document)
            for index, task_id, item, _fingerprint, _attempt, group in tasks:
                if execution == "processes" and index % width in executors:
                    futures[task_id] = executors[index % width].submit(
                        _invoke_campaign_task, fn, item, group
                    )
            # Global collectives use the same devices as the task submeshes.
            # Drain local work before any rank starts exchanging task records.
            for _index, task_id, item, _fingerprint, _attempt, _group in tasks:
                if execution == "serial" or task_id in futures:
                    running = task_ledger[task_id]
                    try:
                        result = (
                            _invoke_campaign_task(fn, item, None)
                            if execution == "serial"
                            else futures[task_id].result()
                        )
                        outcomes[task_id] = (_accept_record(running, result), False)
                    except Exception as error:
                        outcomes[task_id] = (_fail_record(running, error), True)
            for _index, task_id, _item, _fingerprint, _attempt, group in tasks:
                local_record, failed = outcomes.get(task_id, (None, False))
                failure_rank = _failure_source(failed)
                source = (
                    0
                    if group is None
                    else min(device.process_index for device in group.devices.flat)
                )
                if failure_rank < jax.process_count():
                    source = failure_rank
                task_ledger[task_id] = _broadcast_record(local_record, source)
                _write_ledger(path, document)
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)

    records = tuple(_campaign_record(task_ledger[task_id]) for _, task_id, _, _ in declarations)
    results = tuple(record.result for record in records if record.status == "accepted")
    failures = tuple(record.failure for record in records if record.failure is not None)
    return CampaignResult(
        results=results,
        records=records,
        failures=failures,
        ledger_path=str(path),
    )


def deterministic_weighted_reduce(values: Any, weights: Any) -> Any:
    """Return a fixed-order weighted mean of every leaf in a pytree.

    Each leaf has a leading sample axis. Weighted products are sorted along
    that axis independently for every output component before reduction; the
    denominator is likewise sorted. Consequently input/completion
    permutations produce bitwise-identical results while the implementation
    remains expressible with JAX primitives and works under `jax.jit`.

    Args:
        values: Array or pytree of arrays with a common leading sample axis.
        weights: Finite, nonnegative one-dimensional sample weights with
            positive sum.

    Returns:
        The values pytree with its leading sample axis reduced.

    Raises:
        ValueError: Eagerly, for empty/mismatched arrays, invalid weights, or
            complex-valued leaves. Shape checks also occur while tracing.
    """
    weights_array = jnp.asarray(weights)
    if weights_array.ndim != 1 or weights_array.shape[0] == 0:
        raise ValueError("deterministic_weighted_reduce: weights must be a non-empty 1-D array.")
    n_samples = weights_array.shape[0]

    # Host validation is deliberately skipped for traced values; valid calls
    # remain jit-compatible, while eager callers receive precise diagnostics.
    if not isinstance(weights_array, jax.core.Tracer):
        host_weights = local_value(weights_array)
        if not np.all(np.isfinite(host_weights)):
            raise ValueError("deterministic_weighted_reduce: weights must be finite.")
        if np.any(host_weights < 0.0):
            raise ValueError("deterministic_weighted_reduce: weights must be nonnegative.")
        if not float(np.sum(host_weights, dtype=np.float64)) > 0.0:
            raise ValueError("deterministic_weighted_reduce: weights must have a positive sum.")

    leaves = jax.tree_util.tree_leaves(values)
    if not leaves:
        raise ValueError("deterministic_weighted_reduce: values has no array leaves.")

    def reduce_leaf(leaf: Any) -> jax.Array:
        array = jnp.asarray(leaf)
        if array.ndim == 0 or array.shape[0] != n_samples:
            raise ValueError(
                "deterministic_weighted_reduce: every value leaf must have leading "
                f"axis {n_samples}; got shape {array.shape}."
            )
        if jnp.issubdtype(array.dtype, jnp.complexfloating):
            raise ValueError(
                "deterministic_weighted_reduce: complex-valued leaves are unsupported."
            )
        dtype = jnp.result_type(array.dtype, weights_array.dtype, jnp.float32)
        array = array.astype(dtype)
        leaf_weights = weights_array.astype(dtype)
        broadcast_shape = (n_samples,) + (1,) * (array.ndim - 1)
        products = array * jnp.reshape(leaf_weights, broadcast_shape)
        numerator = jnp.sum(jnp.sort(products, axis=0), axis=0, dtype=dtype)
        denominator = jnp.sum(jnp.sort(leaf_weights), dtype=dtype)
        return numerator / denominator

    return jax.tree.map(reduce_leaf, values)


def choose_pattern(workload: str, n_devices: int = 1) -> str:
    """Recommend an execution pattern for a workload (see the module table).

    Args:
        workload: A workload keyword: ``point``/``fim``/``sensitivities``
            (one parameter point), ``robust``/``prior``/``candidates``/
            ``sweep`` (sample batches), ``multistart``/``profile``/``fits``
            (independent runs).
        n_devices: Number of available devices.

    Returns:
        ``"directions"``, ``"samples"``, ``"samples+devices"``, or
        ``"processes"``.

    Raises:
        ValueError: On an unknown workload keyword.
    """
    key = workload.strip().lower()
    if key not in _PATTERNS:
        raise ValueError(
            f"choose_pattern: unknown workload {workload!r}; expected one of "
            f"{sorted(set(_PATTERNS))}."
        )
    pattern = _PATTERNS[key]
    if pattern == "samples" and n_devices > 1:
        return "samples+devices"
    return pattern
