"""Process initialization, global device meshes, and local access to replicas."""

from __future__ import annotations

import contextvars
import functools
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from math import prod
from typing import Any

import jax
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from diff_soup._jax_compat import mesh_utils, multihost_utils

_AXES = frozenset(("data", "model"))
_DEFAULT_XLA_FLAGS = {
    "--xla_gpu_enable_latency_hiding_scheduler": "true",
    "--xla_gpu_all_reduce_combine_threshold_bytes": "16777216",
    # Independent submeshes can compile different kernels at the same time.
    # Distributed autotuning waits for matching programs across all processes.
    "--xla_gpu_shard_autotuning": "false",
}
# The process count and rank an MPI launcher announces: OpenMPI, PMI (MPICH, Slurm's pmi2), PMIx.
_MPI_LAUNCHERS = (
    ("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK"),
    ("PMI_SIZE", "PMI_RANK"),
    ("PMIX_SIZE", "PMIX_RANK"),
)
_debug = False
_decision_mesh = contextvars.ContextVar("diff_soup_decision_mesh", default=None)


def configure() -> None:
    """Add diff-soup's default XLA GPU flags to `XLA_FLAGS`, keeping any flag already set.

    It runs when `diff_soup` is imported and again in `initialize`.
    """
    flags = os.environ.get("XLA_FLAGS", "").split()
    present = {flag.split("=", 1)[0] for flag in flags}
    flags.extend(
        f"{key}={value}" for key, value in _DEFAULT_XLA_FLAGS.items() if key not in present
    )
    os.environ["XLA_FLAGS"] = " ".join(flags)


def initialize(
    *,
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
    debug: bool = False,
    **kwargs: Any,
) -> None:
    """Start JAX's multi-process runtime for a run of several processes.

    Call it before any array is created. Without arguments it detects a Slurm launch with several tasks, `python -m diff_soup.distributed.launch`, or an MPI launch with `DIFFSOUP_COORDINATOR` set, and otherwise does nothing.

    Args:
        coordinator_address: `host:port` of process 0, given together with `num_processes` and `process_id`.
        num_processes: Number of processes in the run.
        process_id: Rank of this process.
        debug: If `True`, check that optimizer decisions agree across processes.
        **kwargs: Passed to `jax.distributed.initialize`.

    Raises:
        ValueError: If only some of `coordinator_address`, `num_processes`, and `process_id` are given.
    """
    global _debug
    configure()
    _debug = bool(debug)
    if jax.distributed.is_initialized():
        _initialize_mgmn_host()
        return
    explicit = (coordinator_address, num_processes, process_id)
    forked = os.environ.get("DIFFSOUP_LAUNCH_COORDINATOR")
    if forked and all(value is None for value in explicit):
        # A process forked by the launcher: one local device, identity from its environment.
        explicit = (
            forked,
            int(os.environ["DIFFSOUP_LAUNCH_SIZE"]),
            int(os.environ["DIFFSOUP_LAUNCH_RANK"]),
        )
        kwargs.setdefault("local_device_ids", [0])
    coordinator_address, num_processes, process_id = explicit
    if any(value is not None for value in explicit):
        if any(value is None for value in explicit):
            raise ValueError(
                "Distributed initialization needs coordinator_address, num_processes, and process_id together."
            )
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
            **kwargs,
        )
    elif _slurm_tasks() > 1:
        # Slurm may expose exactly one GPU per task. Its ordinal inside that
        # visibility mask is zero, even for tasks with a larger node-local rank.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        local_rank = os.environ.get("SLURM_LOCALID")
        pinned = len(visible) == 1 and bool(visible[0]) and local_rank is not None
        try:
            if pinned:
                os.environ["SLURM_LOCALID"] = "0"
            jax.distributed.initialize(**kwargs)
        finally:
            if pinned:
                os.environ["SLURM_LOCALID"] = local_rank
    elif (launched := _mpi_launch()) is not None:
        size, rank, coordinator = launched
        kwargs.setdefault("local_device_ids", [0])
        jax.distributed.initialize(
            coordinator_address=coordinator,
            num_processes=size,
            process_id=rank,
            **kwargs,
        )
    if jax.distributed.is_initialized():
        _initialize_mgmn_host()


def _slurm_tasks() -> int:
    """The task count of the Slurm launch this process belongs to, or 1 outside Slurm.

    ``mpirun`` inside a Slurm job starts the ranks on its own node directly and the ranks on every
    other node through a daemon it launches as a one-task Slurm step, whose environment says
    ``SLURM_NTASKS=1``. The MPI launcher's own world size therefore counts as well, so that every
    rank of one launch takes the same path.
    """
    if "SLURM_JOB_ID" not in os.environ:
        return 1
    tasks = int(os.environ.get("SLURM_NTASKS", "1"))
    for size_name, _ in _MPI_LAUNCHERS:
        size = os.environ.get(size_name, "").strip()
        if size.isdigit():
            tasks = max(tasks, int(size))
    return tasks


def _mpi_launch() -> tuple[int, int, str] | None:
    """``(size, rank, coordinator)`` of an MPI launch to initialize; ``None`` without one.

    An MPI launcher of several processes is initialized only when
    ``DIFFSOUP_COORDINATOR`` names process 0's ``host:port``: without it the
    launch is left to the lane route (``ds.distributed.run``).
    """
    coordinator = os.environ.get("DIFFSOUP_COORDINATOR", "").strip()
    if not coordinator:
        return None
    for size_name, rank_name in _MPI_LAUNCHERS:
        size, rank = os.environ.get(size_name, ""), os.environ.get(rank_name, "")
        if size.strip().isdigit() and rank.strip().isdigit() and int(size) > 1:
            return int(size), int(rank), coordinator
    return None


def _initialize_mgmn_host() -> None:
    """Join the MPI world collectively before any process enters subgroup work."""
    if os.environ.get("DIFFSOUP_MGMN") == "1":
        from diff_soup.linear._cudss._ffi_loader import native_solver

        if not hasattr(native_solver, "mgmn_initialize_mpi"):
            raise RuntimeError("The cuDSS extension must be rebuilt with MGMN support.")
        native_solver.mgmn_initialize_mpi(os.environ.get("DIFFSOUP_MPI_LIBRARY", "libmpi.so.40"))
        from diff_soup.linear._cudss._model_axis._mgmn import finalize_mpi_at_exit

        finalize_mpi_at_exit()


def global_mesh(axis_names: Sequence[str], *, shape: Sequence[int] | None = None) -> Mesh:
    """Build a mesh over every device JAX sees, across all processes.

    A one-axis mesh spans every device, and a two-axis mesh needs `shape`.

    Args:
        axis_names: Distinct names from `("data", "model")`.
        shape: Devices per axis, whose product must equal the device count.

    Returns:
        A `jax.sharding.Mesh` for `Layout(mesh=...)`.
    """
    names = tuple(axis_names)
    if not names or len(set(names)) != len(names) or not set(names) <= _AXES:
        raise ValueError("Mesh axes must be distinct names from ('data', 'model').")
    if shape is None:
        if len(names) != 1:
            raise ValueError("A mesh with data and model axes requires shape=(n_data, n_model).")
        shape = (jax.device_count(),)
    shape = tuple(shape)
    if len(shape) != len(names) or any(not isinstance(n, int) or n < 1 for n in shape):
        raise ValueError("Mesh shape must have one positive integer per axis.")
    if prod(shape) != jax.device_count():
        raise ValueError(
            f"Mesh shape {shape} needs {prod(shape)} devices; JAX has {jax.device_count()}."
        )
    devices = jax.devices()
    axis_types = (AxisType.Auto,) * len(names)
    if (
        devices[0].platform == "gpu"
        and len({getattr(device, "slice_index", 0) for device in devices}) > 1
    ):
        # GPU device IDs encode the topology across nodes. make_mesh rejects
        # multiple slices even though create_device_mesh orders these devices.
        return Mesh(mesh_utils.create_device_mesh(shape, devices), names, axis_types=axis_types)
    return jax.make_mesh(shape, names, axis_types=axis_types, devices=devices)


def mesh_groups(mesh: Mesh, sizes: Sequence[int]) -> tuple[Mesh, ...]:
    """Split a mesh along its `data` axis into consecutive submeshes.

    Pass the groups as `device_groups=` to `ds.inference.checkpointed_map`.

    Args:
        mesh: Mesh to split, usually from `global_mesh`.
        sizes: Number of `data` devices in each group, positive and summing to the axis width.

    Returns:
        One mesh per entry of `sizes`, or the mesh itself when it has no `data` axis.
    """
    sizes = tuple(sizes)
    axis = mesh.axis_names.index("data") if "data" in mesh.axis_names else None
    width = int(mesh.shape["data"]) if axis is not None else 1
    if not sizes or any(not isinstance(n, int) or n < 1 for n in sizes) or sum(sizes) != width:
        raise ValueError(
            f"Submesh sizes must be positive integers summing to {width}; got {sizes}."
        )
    if axis is None:
        return (mesh,)
    offsets = np.cumsum(sizes)[:-1]
    return tuple(
        Mesh(devices, mesh.axis_names, axis_types=(AxisType.Auto,) * len(mesh.axis_names))
        for devices in np.split(mesh.devices, offsets, axis=axis)
    )


def owns(mesh: Mesh) -> bool:
    """Return whether any device of `mesh` belongs to this process.

    Args:
        mesh: Mesh to test.

    Returns:
        `True` when this process holds a device of `mesh`.
    """
    return any(device.process_index == jax.process_index() for device in mesh.devices.flat)


def local_value(value: Any, dtype=None) -> np.ndarray:
    """Read a value as a NumPy array on this process.

    A global array that this process cannot fully address must be replicated, and its first local shard is read.

    Args:
        value: Array or array-like to read.
        dtype: Optional NumPy dtype of the result.

    Returns:
        A host `numpy.ndarray`.

    Raises:
        ValueError: If `value` is a partitioned global array.
    """
    if isinstance(value, jax.Array) and not value.is_fully_addressable:
        if not value.is_fully_replicated or not value.addressable_shards:
            raise ValueError(
                "A partitioned global array cannot be read as a local value; select addressable_shards or use process_allgather."
            )
        value = value.addressable_shards[0].data
    return np.asarray(value, dtype=dtype)


def from_local_data(data: Any, mesh: Mesh, spec: PartitionSpec, *, global_shape=None) -> jax.Array:
    """Assemble a global array from the piece of it each process holds.

    Args:
        data: This process's local piece.
        mesh: Mesh the array is placed on.
        spec: `PartitionSpec` of the array over `mesh`.
        global_shape: Shape of the global array, inferred from the local pieces when `None`.

    Returns:
        A `jax.Array` sharded by `NamedSharding(mesh, spec)`.
    """
    return jax.make_array_from_process_local_data(NamedSharding(mesh, spec), data, global_shape)


@contextmanager
def _decision_scope(mesh):
    """Limit optimizer decision checks to the processes running the current task."""
    token = _decision_mesh.set(mesh)
    try:
        yield
    finally:
        _decision_mesh.reset(token)


@functools.lru_cache(maxsize=64)
def _collective(devices: tuple, kind: str) -> tuple:
    """The program that combines one value per device of ``devices``, built once per device set.

    ``kind`` is ``"same"`` (whether every value is equal) or ``"any"`` (the
    largest value). Returns ``(sharding, program, local)``: the sharding of the
    stacked values over a one-axis ``data`` mesh of ``devices``, the jitted
    ``shard_map`` that reduces them to one replicated value, and the devices of
    ``devices`` this process holds. A later call with the same devices and kind
    returns the same program, so a decision of the same shape lowers nothing.
    """
    mesh = Mesh(np.asarray(devices, dtype=object), ("data",), axis_types=(AxisType.Auto,))
    if kind == "same":

        def combine(x):
            return (jax.lax.pmin(x, "data") == jax.lax.pmax(x, "data")).all()

    elif kind == "any":

        def combine(x):
            return jax.lax.pmax(x, "data")

    else:
        raise ValueError(f"a collective is 'same' or 'any'; got {kind!r}.")
    program = jax.jit(
        jax.shard_map(
            combine,
            mesh=mesh,
            in_specs=PartitionSpec("data"),
            out_specs=PartitionSpec(),
            check_vma=True,
        )
    )
    local = tuple(device for device in devices if device.process_index == jax.process_index())
    return NamedSharding(mesh, PartitionSpec("data")), program, local


def _combined(devices: Any, kind: str, value: np.ndarray) -> Any:
    """Every device's ``value`` of ``devices``, stacked and reduced by the ``kind`` collective."""
    devices = tuple(np.asarray(devices, dtype=object).reshape(-1).tolist())
    sharding, program, local = _collective(devices, kind)
    buffers = [jax.device_put(value[None], device) for device in local]
    placed = jax.make_array_from_single_device_arrays(
        (len(devices), *value.shape), sharding, buffers
    )
    return program(placed)


def assert_decision(decision: Any, *, label: str = "optimizer decision") -> None:
    """In debug mode, verify the active optimizer group agrees on a decision.

    The collective of a group's devices is compiled once per device set and
    decision shape; later decisions reuse it.
    """
    if not _debug or jax.process_count() == 1:
        return
    group = _decision_mesh.get()
    if group is None:
        multihost_utils.assert_equal(local_value(decision), fail_message=label)
        return
    if len({device.process_index for device in group.devices.flat}) == 1:
        return
    value = np.asarray(local_value(decision), dtype=np.int32)
    if not bool(local_value(_combined(group.devices, "same", value))):
        raise AssertionError(label)


def any_process(flag: Any) -> bool:
    """Return whether `flag` is true on any process.

    It is a collective, so every process taking part must call it. In one process it returns `flag`.

    Args:
        flag: Boolean, or a scalar array read with `local_value`.

    Returns:
        `True` when `flag` is true on at least one process.
    """
    flag = bool(local_value(flag))
    if jax.process_count() == 1:
        return flag
    group = _decision_mesh.get()
    if group is None:
        devices = np.asarray(jax.devices(), dtype=object)
    else:
        devices = group.devices
        if len({device.process_index for device in devices.flat}) == 1:
            return flag
    return bool(local_value(_combined(devices, "any", np.asarray(flag, dtype=np.int32))))


def assert_decision_value(decision: Any, *, label: str = "optimizer decision") -> bool:
    """Read a scalar decision on the host after checking every process agrees on it.

    The same as [assert_decision][diff_soup._distributed.assert_decision] followed by
    ``bool(local_value(decision))``, for the accept, reject, and stop branches of an
    optimizer loop that must take the same path on every rank.
    """
    assert_decision(decision, label=label)
    return bool(local_value(decision))


@contextmanager
def run(study: Any, **options: Any) -> Iterator[Any]:
    """Lower a study over the processes of a run, and close it when the block ends.

    Use it as `with ds.distributed.run(study) as lowered:`. In a run of several processes, process 0 runs the block and the others evaluate their experiments until it ends, so write results inside the block.

    Args:
        study: The [Study][diff_soup.inference.Study] to lower.
        **options: Passed to `ds.lower`, for example `objective=` or `layout=`.

    Yields:
        The lowered study.

    Raises:
        TypeError: If `study` is not a `Study`.
        SystemExit: With status 0, on every process above 0 once process 0 leaves the block.
    """
    from diff_soup import lower

    if getattr(type(study), "_ds_lower", None) is None:
        raise TypeError(
            f"ds.distributed.run lowers a Study; got {type(study).__name__}. Declare the "
            "experiments in a diff_soup.inference.Study and pass it."
        )
    lowered = lower(study, **options)
    try:
        yield lowered
    finally:
        lowered.close()


def key(seed: int) -> jax.Array:
    """Return `jax.random.key(seed)`.

    Pass the same seed on every process so they draw the same random numbers.

    Args:
        seed: Integer seed.

    Returns:
        A typed JAX random key.
    """
    return jax.random.key(seed)


def is_primary() -> bool:
    """Return whether this process is process 0.

    Returns:
        `True` on process 0.
    """
    return jax.process_index() == 0


configure()
