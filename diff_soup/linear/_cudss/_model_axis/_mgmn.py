"""cuDSS communicator setup for one factorization across the processes of a model group."""

from __future__ import annotations

import atexit
import hashlib
import os

import jax
import numpy as np
from jax.sharding import AxisType, Mesh

from diff_soup._errors import ConfigurationError
from diff_soup.linear._cudss._ffi_loader import _wheel_library, detect_cuda_line, native_solver

_groups = {}


def model_mesh(mesh):
    """Select this process's factorization group, preserving model rank order."""
    axis = mesh.axis_names.index("model")
    rows = np.moveaxis(mesh.devices, axis, -1).reshape((-1, mesh.shape["model"]))
    owned = [row for row in rows if any(d.process_index == jax.process_index() for d in row)]
    if len(owned) != 1 or len({d.process_index for d in owned[0]}) != len(owned[0]):
        raise ConfigurationError(
            "MGMN requires one GPU per process and one model group per process."
        )
    return Mesh(owned[0], ("model",), axis_types=(AxisType.Auto,))


def prepare(mesh):
    """Initialize a model communicator once, before its first compiled solve.

    cuDSS's NCCL communication layer (``libcudss_commlayer_nccl.so*`` of the
    cuDSS wheel) uses Open MPI for host communication. Both JAX
    and MPI must have the same Slurm ranks; the launcher enables PMI2 and
    makes ``libmpi.so.40`` available. GPU communication uses the NCCL wheel
    already loaded by JAX.

    The processes of the group agree on the NCCL id of its communicator over
    MPI on the host (``broadcast_from_root``), so an analysis may run while a
    JAX program is traced. A group is set up at its first call in each
    process; every process of the group runs the same program, so each one
    reaches that call in the same order.
    """
    group = model_mesh(mesh)
    ranks = tuple(device.process_index for device in group.devices.flat)
    identity = tuple(device.id for device in group.devices.flat)
    key = (ranks, identity)
    if key in _groups:
        return _groups[key]
    if any(device.platform != "gpu" for device in group.devices.flat):
        raise ConfigurationError("cuDSS MGMN needs CUDA devices on its model axis.")
    if not all(hasattr(native_solver, name) for name in ("mgmn_initialize", "mgmn_broadcast")):
        raise ConfigurationError("The cuDSS extension must be rebuilt with MGMN support.")
    nccl = _wheel_library("nvidia.nccl", "libnccl.so.2")
    line = detect_cuda_line().removeprefix("jax_cuda").removesuffix("_plugin")
    shim = _wheel_library(f"nvidia.cu{line}", "libcudss_commlayer_nccl.so*")
    if nccl is None or shim is None:
        raise ConfigurationError(
            "MGMN needs the NVIDIA NCCL wheel and cuDSS NCCL communication layer."
        )
    group_id = int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:4], "big") & 0x7FFFFFFF
    group_id = group_id or 1
    unique = _shared_unique_id(group_id, ranks, str(nccl))
    native_solver.mgmn_initialize(
        group_id,
        list(ranks),
        jax.process_index(),
        unique,
        str(nccl),
        _mpi_library(),
        str(shim),
    )
    _groups[key] = group_id
    return group_id


#: The length of the error text of the NCCL id exchange.
_ERROR_BYTES = 512


def _shared_unique_id(group_id: int, ranks: tuple[int, ...], nccl: str) -> bytes:
    """The NCCL id the group's first process creates, on every process of the group.

    The first process sends a failure flag, the 128-byte id and its error
    text (empty on success) in one broadcast, so a failure to create the id
    is raised on every process of the group instead of leaving the others
    waiting for the communicator.

    Raises:
        ConfigurationError: On every process of the group, when the first
            process could not create the id.
    """
    payload = bytes(1 + 128 + _ERROR_BYTES)
    if jax.process_index() == ranks[0]:
        try:
            payload = b"\0" + bytes(native_solver.mgmn_unique_id(nccl)) + bytes(_ERROR_BYTES)
        except Exception as error:  # every process raises it below
            text = str(error).encode("utf-8")[:_ERROR_BYTES]
            payload = b"\1" + bytes(128) + text.ljust(_ERROR_BYTES, b"\0")
    received = broadcast_from_root(group_id, ranks, payload)
    if received[0]:
        text = received[129:].rstrip(b"\0").decode("utf-8", "replace")
        raise ConfigurationError(
            f"cuDSS MGMN: JAX process {ranks[0]}, the first of the model group, could not create "
            f"the NCCL communicator id: {text}"
        )
    return received[1:129]


def broadcast_from_root(group_id: int, ranks: tuple[int, ...], data: bytes) -> bytes:
    """The ``data`` of the group's first process, returned on every process of the group.

    The bytes travel over the group's MPI host communicator (``MPI_Bcast``),
    which the native extension makes at the group's first call and keeps for
    its cuDSS factors. This is a host collective, independent of JAX, so it
    runs while a JAX program is traced. Every process in ``ranks`` makes the
    same sequence of these calls; a process outside ``ranks`` makes none.

    Args:
        group_id: The id of the group, the same in every process of it.
        ranks: The JAX process indices of the group (its MPI world ranks),
            the first being the process whose ``data`` every process takes.
        data: This process's bytes, of the same length in every process.

    Returns:
        The first process's ``data``.
    """
    received = native_solver.mgmn_broadcast(
        int(group_id), list(ranks), jax.process_index(), bytes(data), _mpi_library()
    )
    finalize_mpi_at_exit()
    return received


_finalize_registered = False


def finalize_mpi_at_exit() -> None:
    """Finalize, at interpreter exit, the MPI world this library initialized.

    Registered once, after the native extension has initialized or joined
    MPI. At exit the extension destroys the cuDSS objects of every MGMN
    factor and calls ``MPI_Finalize``, so that ``mpirun`` reports a run whose
    processes all returned 0 as a success. It finalizes only an MPI it
    initialized itself (``MPI_Init_thread``) and not yet finalized: an MPI
    that mpi4py, or JAX's MPI collectives for CPU devices, initialized is
    left to them, which finalize what they initialized.

    The handler runs before JAX's own exit handlers, which were registered
    when JAX was imported: JAX later destroys its backends, and with them
    the programs that hold the factors, and then shuts down its distributed
    client. Because the factors' cuDSS objects are destroyed before
    ``MPI_Finalize``, that later teardown makes no MPI call; JAX's
    distributed client talks to its coordination service and not over MPI,
    so finalizing MPI before its shutdown is safe.
    """
    global _finalize_registered
    if _finalize_registered or not hasattr(native_solver, "mgmn_finalize_mpi"):
        return
    atexit.register(native_solver.mgmn_finalize_mpi)
    _finalize_registered = True


def _mpi_library() -> str:
    """The Open MPI library the native extension loads (``DIFFSOUP_MPI_LIBRARY``)."""
    return os.environ.get("DIFFSOUP_MPI_LIBRARY", "libmpi.so.40")
