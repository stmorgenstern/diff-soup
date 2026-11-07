"""The environment variables diff_soup reads, and what importing it changes in the process.

Every variable diff_soup reads is declared here as a [Variable][diff_soup._environment.Variable]
with its default, and every change ``import diff_soup`` makes to the process is made
here, when the package imports this module before anything else. The installation
page lists the same variables and changes for users.

Importing diff_soup makes three changes, in this order:

1. ``XLA_FLAGS`` gains the flags of [XLA_GPU_FLAGS][diff_soup._environment.XLA_GPU_FLAGS]
   that it does not already set ([add_xla_flags][diff_soup._environment.add_xla_flags],
   public as ``ds.distributed.configure``).
2. ``JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS`` is set to ``1`` when unset, before JAX is
   imported, and JAX's ``jax_use_simplified_jaxpr_constants`` flag is turned on, so a
   compiled program takes the arrays it closes over as arguments
   (``diff_soup._jax_constants``, which also repairs JAX when it was imported first).
3. JAX's custom-call builder is wrapped so that a typed-FFI call carries its
   configuration on the op itself (``diff_soup._jax_typed_ffi``), unless
   ``DIFF_SOUP_INHERENT_FFI_CONFIG=0``.

Loading the cuDSS backend prepends the NVIDIA wheels' library directories to
``LD_LIBRARY_PATH`` for child processes
([prepend_library_path][diff_soup._environment.prepend_library_path], called by
``diff_soup.linear._cudss._ffi_loader``).

The cuDSS extension reads five variables of its own when a handle starts, each leaving
cuDSS's default in place when unset (``_internal/cudss_ffi/native/single_solve.cpp``):
``DIFFSOUP_CUDSS_HOST_NTHREADS`` (threads of the host side of the analysis),
``DIFFSOUP_CUDSS_ND_NLEVELS`` and ``DIFFSOUP_CUDSS_ND_UBFACTOR`` (the nested-dissection
depth and balance), ``DIFFSOUP_CUDSS_HYBRID_EXECUTE`` (hybrid host and device execution),
and the diagnostics ``DIFF_SOUP_CUDSS_DEBUG_PTRS``, ``DIFF_SOUP_CUDSS_DEBUG_MAX_CALLS`` and
``DIFF_SOUP_CUDSS_CUDA_EVENTS``.

This module imports nothing from diff_soup and JAX only inside the function that runs
at import, after ``JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS`` is in place.
"""

from __future__ import annotations

import os
import sys
from typing import NamedTuple


class Variable(NamedTuple):
    """An environment variable diff_soup reads.

    Attributes:
        name: The variable's name.
        default: Its value when the variable is unset or blank; ``None`` for no value.
    """

    name: str
    default: str | None = None

    def get(self) -> str | None:
        """Return the variable's value, or its default when it is unset or blank."""
        value = os.environ.get(self.name)
        return value if value is not None and value.strip() else self.default


INHERENT_FFI_CONFIG = Variable("DIFF_SOUP_INHERENT_FFI_CONFIG", "1")
"""``0`` leaves JAX's custom-call builder unwrapped; read once, when diff_soup is imported."""

COORDINATOR = Variable("DIFFSOUP_COORDINATOR")
"""``host:port`` of process 0: ``ds.distributed.initialize()`` initializes an MPI launch of
several processes only when it is set."""

MGMN = Variable("DIFFSOUP_MGMN", "0")
"""``1`` in every process runs a ``model`` axis whose GPUs belong to several processes in
cuDSS's multi-node mode, and has the launcher start the ranks with ``srun --mpi=pmi2``."""

MPI_LIBRARY = Variable("DIFFSOUP_MPI_LIBRARY", "libmpi.so.40")
"""The Open MPI library the cuDSS extension loads in the multi-node mode."""

CACHE_DIR = Variable("DIFFSOUP_CACHE_DIR")
"""The directory window initializations persist under. Unset, it is ``diff_soup`` beside
JAX's persistent compilation cache when one is configured, else ``~/.cache/diff_soup``,
which a run of several processes refuses."""

DEADLINE_SECONDS = Variable("DIFF_SOUP_DEADLINE_SECONDS", "600")
"""The shortest time process 0 of a lane run waits for any reply, in seconds."""

PREPARE_DEADLINE_SECONDS = Variable("DIFF_SOUP_PREPARE_DEADLINE_SECONDS", "21600")
"""The time process 0 of a lane run waits for every process to prepare its lanes, in seconds."""

NODES = Variable("DIFFSOUP_NODES", "1")
"""The nodes ``python -m diff_soup.distributed.launch`` runs on under Slurm."""

GPUS_PER_NODE = Variable("DIFFSOUP_GPUS_PER_NODE", "2")
"""The ranks, one per GPU, the launcher starts on each node under Slurm."""

CPUS_PER_PROCESS = Variable("DIFFSOUP_CPUS_PER_PROCESS", "4")
"""The CPUs the launcher gives each rank under Slurm."""

PROCESSES = Variable("DIFFSOUP_PROCESSES")
"""The process count of a launch the launcher forks on this host; required on a CPU-only
host, where each process gets one host device. Unset, one process runs per visible GPU."""

LAUNCH_ROLE = Variable("DIFFSOUP_LAUNCH_ROLE")
"""Set to ``rank`` by the launcher in the processes it starts."""

LAUNCH_COORDINATOR = Variable("DIFFSOUP_LAUNCH_COORDINATOR")
"""The local ``host:port`` of process 0, set by the launcher in the processes it forks."""

LAUNCH_SIZE = Variable("DIFFSOUP_LAUNCH_SIZE")
"""The process count, set by the launcher in the processes it forks."""

LAUNCH_RANK = Variable("DIFFSOUP_LAUNCH_RANK")
"""The process's rank, set by the launcher in the processes it forks."""

MGMN_GPU_MASK = Variable("DIFFSOUP_MGMN_GPU_MASK")
"""The allocation's ``CUDA_VISIBLE_DEVICES``, passed by the launcher to the ranks of a
one-node multi-node-mode step, each of which keeps the GPU of its local rank."""

HOIST_CONSTANTS = "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS"
"""JAX's variable for constant hoisting, which JAX reads when it is imported."""

XLA_GPU_FLAGS: dict[str, str] = {
    "--xla_gpu_enable_latency_hiding_scheduler": "true",
    # 16 MiB (ours).
    "--xla_gpu_all_reduce_combine_threshold_bytes": "16777216",
    # Independent submeshes can compile different kernels at the same time, and
    # distributed autotuning waits for matching programs across all processes.
    "--xla_gpu_shard_autotuning": "false",
}
"""The XLA GPU flags diff_soup adds to ``XLA_FLAGS`` (ours)."""


def add_xla_flags() -> None:
    """Add diff-soup's default XLA GPU flags to `XLA_FLAGS`, keeping any flag already set.

    It runs when `diff_soup` is imported and again in `initialize`.
    """
    flags = os.environ.get("XLA_FLAGS", "").split()
    present = {flag.split("=", 1)[0] for flag in flags}
    flags.extend(f"{key}={value}" for key, value in XLA_GPU_FLAGS.items() if key not in present)
    os.environ["XLA_FLAGS"] = " ".join(flags)


def prepend_library_path(directories: list[str]) -> None:
    """Put ``directories`` first on ``LD_LIBRARY_PATH``, keeping every other entry once.

    It affects child processes only: this process resolved its search path at start-up.
    """
    existing = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(p for p in directories + existing if p))


def _configure_process() -> None:
    """Make the changes the module docstring lists, in its order."""
    jax_imported_first = "jax" in sys.modules
    add_xla_flags()
    os.environ.setdefault(HOIST_CONSTANTS, "1")
    from diff_soup import _jax_constants, _jax_typed_ffi

    _jax_constants.turn_on(jax_imported_first=jax_imported_first)
    if INHERENT_FFI_CONFIG.get() != "0":
        _jax_typed_ffi.install()


_configure_process()
