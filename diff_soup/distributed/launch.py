"""Run a script on every process of an allocation, one process per GPU.

Use ``python -m diff_soup.distributed.launch script.py [arguments]``. The script
calls ``ds.distributed.initialize()`` before it creates arrays; the launcher
supplies the process identity that call reads.

Where the processes run depends on the environment:

* Under Slurm with ``DIFFSOUP_MGMN=1``, one ``srun --mpi=pmi2`` step launches
  the ranks so JAX and MPI share Slurm's rank assignment (cuDSS's distributed
  solver uses Open MPI for host buffers; set ``DIFFSOUP_MPI_LIBRARY`` to the
  ``libmpi.so.40`` path when it is not on the loader path).
* Under Slurm across several nodes, one ``srun`` step launches one rank per
  GPU on every node.
* Inside a one-task Slurm step that already owns a node's GPUs, and outside
  Slurm, the launcher forks one process per visible GPU on this host and
  coordinates them over a local port.

Variables, with their defaults: ``DIFFSOUP_NODES`` (1),
``DIFFSOUP_GPUS_PER_NODE`` (2), ``DIFFSOUP_CPUS_PER_PROCESS`` (4),
``DIFFSOUP_MGMN`` (0). ``DIFFSOUP_PROCESSES`` sets the process count of a
forked launch explicitly; it is required on a CPU-only host, where each
process gets one host device. With ``JAX_COMPILATION_CACHE_DIR`` set, every
rank compiles into its own ``rank-<index>`` subdirectory, because the
persistent cache writes without a lock and a rank reading an entry another
rank is still writing sees a truncated stream.
"""

from __future__ import annotations

import os
import runpy
import shutil
import socket
import subprocess
import sys
import time

__all__ = ["main", "rank_cache_dir"]

_ROLE = "DIFFSOUP_LAUNCH_ROLE"
_COORDINATOR = "DIFFSOUP_LAUNCH_COORDINATOR"
_SIZE = "DIFFSOUP_LAUNCH_SIZE"
_RANK = "DIFFSOUP_LAUNCH_RANK"


def rank_cache_dir(base: str | None, rank: int) -> str | None:
    """Return rank ``rank``'s compilation cache directory under ``base``, or ``None`` without one."""
    if not base:
        return None
    return os.path.join(base, f"rank-{rank}")


def _settings() -> dict[str, str]:
    """Read the launch variables with their defaults."""
    return {
        "nodes": os.environ.get("DIFFSOUP_NODES", "1"),
        "gpus_per_node": os.environ.get("DIFFSOUP_GPUS_PER_NODE", "2"),
        "cpus_per_process": os.environ.get("DIFFSOUP_CPUS_PER_PROCESS", "4"),
        "mgmn": os.environ.get("DIFFSOUP_MGMN", "0"),
    }


def _restore_allocation_nodelist(nodes: int) -> None:
    """Widen a job step's node list back to the whole allocation for a multi-node launch."""
    job = os.environ.get("SLURM_JOB_ID")
    if nodes <= 1 or not job:
        return
    shown = subprocess.run(
        ["scontrol", "show", "job", job, "-o"], capture_output=True, text=True, check=True
    ).stdout
    nodelist = next(
        (field[len("NodeList=") :] for field in shown.split() if field.startswith("NodeList=")),
        None,
    )
    if nodelist:
        os.environ["SLURM_JOB_NODELIST"] = nodelist
        os.environ["SLURM_NODELIST"] = nodelist
        os.environ["SLURM_NNODES"] = str(nodes)


def _srun(arguments: list[str], script: list[str]) -> None:
    """Replace this process with one Slurm step running the ranks."""
    command = [
        "srun",
        "--overlap",
        "--cpu-bind=none",
        *arguments,
        sys.executable,
        "-m",
        "diff_soup.distributed.launch",
        *script,
    ]
    os.environ[_ROLE] = "rank"
    os.execvp(command[0], command)


def _visible_gpus() -> list[str]:
    """The GPUs this host exposes: the visibility mask, else the driver's list."""
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is not None:
        return [device for device in mask.split(",") if device]
    if shutil.which("nvidia-smi") is None:
        return []
    listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if listing.returncode != 0:
        return []
    return [str(index) for index, line in enumerate(listing.stdout.splitlines()) if line.strip()]


def _fork(script: list[str]) -> int:
    """Fork one process per local GPU (or per requested CPU process) and wait for them."""
    cpu_only = os.environ.get("JAX_PLATFORMS", "").lower() == "cpu"
    devices = [] if cpu_only else _visible_gpus()
    if len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES lists a GPU twice; give each process its own GPU.")
    requested = os.environ.get("DIFFSOUP_PROCESSES")
    if requested is not None:
        count = int(requested)
        if count < 1:
            raise ValueError("DIFFSOUP_PROCESSES must be at least 1.")
        if devices and count > len(devices):
            raise ValueError(
                f"DIFFSOUP_PROCESSES={count} exceeds the {len(devices)} visible GPUs; "
                "one process runs per GPU."
            )
    elif devices:
        count = len(devices)
    else:
        raise ValueError(
            "No GPU is visible to this host. Set DIFFSOUP_PROCESSES to the number of "
            "CPU processes to fork (with JAX_PLATFORMS=cpu), or launch inside a GPU allocation."
        )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"
    base = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SLURM_") and "proxy" not in key.lower()
    }
    children = []
    try:
        for rank in range(count):
            environment = dict(base)
            environment.update(
                {
                    _ROLE: "rank",
                    _COORDINATOR: coordinator,
                    _SIZE: str(count),
                    _RANK: str(rank),
                }
            )
            if devices:
                environment["CUDA_VISIBLE_DEVICES"] = devices[rank]
            else:
                flags = environment.get("XLA_FLAGS", "").split()
                flags = [
                    flag
                    for flag in flags
                    if not flag.startswith("--xla_force_host_platform_device_count")
                ]
                flags.append("--xla_force_host_platform_device_count=1")
                environment["XLA_FLAGS"] = " ".join(flags)
            children.append(
                subprocess.Popen(
                    [sys.executable, "-m", "diff_soup.distributed.launch", *script],
                    env=environment,
                )
            )
        while any(child.poll() is None for child in children):
            failed = next((child.returncode for child in children if child.returncode), None)
            if failed:
                return failed
            time.sleep(0.1)
        return max(child.returncode for child in children)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def _run_rank(script: list[str]) -> None:
    """Run the script in this rank: select its GPU and cache directory, then hand over."""
    mask = os.environ.get("DIFFSOUP_MGMN_GPU_MASK")
    if mask is not None:
        devices = mask.split(",")
        local_rank = int(os.environ["SLURM_LOCALID"])
        if local_rank >= len(devices) or len(set(devices)) != len(devices):
            raise ValueError("The Slurm step needs a distinct allocated GPU for every local rank.")
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get(_RANK, "0")))
    cache = rank_cache_dir(os.environ.get("JAX_COMPILATION_CACHE_DIR"), rank)
    if cache is not None:
        os.makedirs(cache, exist_ok=True)
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache
        import jax

        jax.config.update("jax_compilation_cache_dir", cache)
    sys.argv = list(script)
    try:
        runpy.run_path(script[0], run_name="__main__")
    finally:
        # Every rank leaves the distributed service at the same point, after
        # the script and after its device work has drained, so the shutdown
        # barrier sees all of them whether the script returned or raised.
        import jax

        if jax.distributed.is_initialized():
            from diff_soup._jax_compat import multihost_utils

            multihost_utils.sync_global_devices("diff_soup.distributed.launch.exit")
            jax.distributed.shutdown()


def main(argv: list[str] | None = None) -> int:
    """Launch ``argv[0]`` with its arguments on every process of the allocation."""
    script = list(sys.argv[1:] if argv is None else argv)
    if not script:
        raise SystemExit("usage: python -m diff_soup.distributed.launch script.py [arguments]")
    if os.environ.get(_ROLE) == "rank":
        _run_rank(script)
        return 0
    settings = _settings()
    nodes = int(settings["nodes"])
    gpus_per_node = int(settings["gpus_per_node"])
    cpus = settings["cpus_per_process"]
    in_slurm = bool(os.environ.get("SLURM_JOB_ID"))
    if in_slurm:
        _restore_allocation_nodelist(nodes)
    if in_slurm and settings["mgmn"] == "1":
        # cuDSS allocates distributed factors outside JAX's memory pool.
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        if nodes == 1:
            os.environ["DIFFSOUP_MGMN_GPU_MASK"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            _srun(
                [
                    "--mpi=pmi2",
                    "--nodes=1",
                    f"--ntasks={gpus_per_node}",
                    f"--gres=gpu:{gpus_per_node}",
                    f"--cpus-per-task={cpus}",
                ],
                script,
            )
        _srun(
            [
                "--mpi=pmi2",
                f"--nodes={nodes}",
                f"--ntasks={nodes * gpus_per_node}",
                f"--ntasks-per-node={gpus_per_node}",
                "--gpus-per-task=1",
                f"--cpus-per-task={cpus}",
            ],
            script,
        )
    if in_slurm and (nodes > 1 or int(os.environ.get("SLURM_NTASKS", "1")) > 1):
        # Set NCCL_SOCKET_IFNAME and InfiniBand variables to the values your
        # cluster administrators supply; the package chooses no network interface.
        _srun(
            [
                f"--nodes={nodes}",
                f"--ntasks={nodes * gpus_per_node}",
                f"--ntasks-per-node={gpus_per_node}",
                "--gpus-per-task=1",
                f"--cpus-per-task={cpus}",
            ],
            script,
        )
    return _fork(script)


if __name__ == "__main__":
    raise SystemExit(main())
