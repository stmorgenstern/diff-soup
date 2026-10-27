"""The processes of a run and the messages between them, over MPI.

A run of several processes is one MPI world launched with one process per GPU
(``srun --mpi=pmi2 --kill-on-bad-exit=1 -n K --gpus-per-task=1 python -m mpi4py
script.py``). Process 0 coordinates and evaluates lanes; processes 1 .. K-1
evaluate lanes. [world][diff_soup.inference.engine._transport.world]
(``inf.world()``) answers which process this is and how many there are, before
or after ``ds.lower``; without ``mpi4py``, or outside an MPI launch, the world
is this one process and nothing here is imported. ``mpi4py`` is imported here
only, and only when an MPI launch is detected.

The protocol, in the order a run uses it (``MESSAGES``):

| message | payload | what a worker does |
|---|---|---|
| ``PLAN_CHECK`` | the plan's fingerprint | compares it with its own; every process refuses a mismatch |
| ``PLACE`` | lane id to process | keeps its own lanes and builds their inputs |
| ``PREPARE`` | modes, the cut, the memory share, the compiling process of every bucket | compiles its buckets' programs in two phases and replies with its records and refusals |
| ``EVALUATE`` | the request, its points and the fit it belongs to | evaluates its own units, sums its lanes over the level tables and replies with the sums and the per-lane numbers |
| ``PRIVATE_SAVE``, ``PRIVATE_LOAD`` | a directory and an attempt id | nothing yet: the lanes hold no private coordinates |
| ``SHUTDOWN`` | nothing | frees its device memory, runs its exit handlers, acknowledges, finalizes MPI and exits 0 |

Headers are small Python objects sent pickled (``bcast`` down, a length and a
byte buffer up); arrays travel as numpy buffers (``Bcast`` down, ``Send``
up) received into fresh ``np.empty`` buffers. Process 0 waits for every reply
with non-blocking receives polled against a deadline: MPI receives have no
timeout and a wedged device never answers, so a reply later than the deadline
aborts the whole run (``MPI_Abort``) and the run resumes from its checkpoint.
"""

from __future__ import annotations

import os
import pickle
import sys
import time
import warnings
from typing import Any, NamedTuple

import numpy as np

from diff_soup._errors import ConfigurationError

MESSAGES: tuple = (
    "PLAN_CHECK",
    "PLACE",
    "PREPARE",
    "EVALUATE",
    "PRIVATE_SAVE",
    "PRIVATE_LOAD",
    "SHUTDOWN",
)
"""The messages of the protocol, in the order a run uses them."""

_TAG_LENGTH = 7101
_TAG_HEADER = 7102
_TAG_ARRAY = 7103
_ABORT_CODE = 3
_POLL_SECONDS = 0.0005


def _seconds(name: str, default: float) -> float:
    text = os.environ.get(name)
    if text is None or not text.strip():
        return float(default)
    try:
        value = float(text)
    except ValueError:
        raise ConfigurationError(
            f"{name}={text!r} is not a number of seconds; give a positive number."
        ) from None
    if not value > 0.0:
        raise ConfigurationError(f"{name}={text!r}; give a positive number of seconds.")
    return value


DEADLINE_FLOOR_SECONDS: float = _seconds("DIFF_SOUP_DEADLINE_SECONDS", 600.0)
"""The shortest time process 0 waits for any reply, in seconds.

600 unless the environment variable ``DIFF_SOUP_DEADLINE_SECONDS`` gives
another; process 0 reads it at every wait, so setting it in process 0 changes
the next wait.
"""

DEADLINE_FACTOR: float = 10.0
"""A reply to ``EVALUATE`` is due within this factor times the longest time one
evaluation of a lane took so far (the study's ledger) times the most units any
process evaluates in the request, and never sooner than ``DEADLINE_FLOOR_SECONDS``;
before any lane has a measured time, within the deadline of a prepare."""

PREPARE_DEADLINE_SECONDS: float = _seconds("DIFF_SOUP_PREPARE_DEADLINE_SECONDS", 21600.0)
"""The time process 0 waits for every process's reply to ``PREPARE``, in seconds:
21,600 unless ``DIFF_SOUP_PREPARE_DEADLINE_SECONDS`` gives another."""


def evaluation_deadline(lane_seconds: float, units: int) -> float:
    """The deadline of an evaluation of at most ``units`` units per process, in seconds.

    Before any lane has a measured time (``lane_seconds`` is 0), nothing bounds
    how long a correct first evaluation takes on a large instance, so the
    deadline is that of a prepare (``prepare_deadline()``). Once a time is
    measured, it is ``DEADLINE_FACTOR`` times that time times ``units``, never
    below ``DEADLINE_FLOOR_SECONDS``.
    """
    if not float(lane_seconds) > 0.0:
        return prepare_deadline()
    measured = DEADLINE_FACTOR * float(lane_seconds) * max(int(units), 1)
    return max(float(DEADLINE_FLOOR_SECONDS), measured)


def prepare_deadline() -> float:
    """The deadline of a prepare, in seconds; never below ``DEADLINE_FLOOR_SECONDS``."""
    return max(float(PREPARE_DEADLINE_SECONDS), float(DEADLINE_FLOOR_SECONDS))


def reply_deadline() -> float:
    """The deadline of any other reply, in seconds."""
    return float(DEADLINE_FLOOR_SECONDS)


class World(NamedTuple):
    """The processes of this run: this process's rank and how many there are.

    Attributes:
        rank: This process's rank: 0 for the process that continues the
            script past ``ds.lower`` and coordinates the run, 1 .. ``size - 1``
            for the others.
        size: The number of processes; 1 outside an MPI launch.
    """

    rank: int
    size: int


# This process's world and its MPI communicator (None for a world of one process), once found.
_WORLD: list = []


def _launched_size() -> int:
    """The process count an MPI launcher announced in the environment; 1 without one."""
    for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MPI_LOCALNRANKS"):
        text = os.environ.get(name)
        if text and text.strip().isdigit():
            return int(text)
    return 1


def _jax_distributed() -> bool:
    """Whether JAX's own multi-process runtime is initialized (the global-mesh route)."""
    jax = sys.modules.get("jax")
    if jax is None:
        return False
    try:
        return bool(jax.distributed.is_initialized())
    except AttributeError:
        return False


def world() -> World:
    """Return this process's rank and the number of processes in the run.

    A process not launched under MPI, or one using `jax.distributed`, is process 0 of 1. Launch
    several processes with `python -m mpi4py`, which needs mpi4py installed.

    Returns:
        A named tuple `(rank, size)`.
    """
    if _WORLD:
        found = _WORLD[0][0]
        if found.size > 1 and _jax_distributed():
            raise ConfigurationError(
                f"jax.distributed was initialized after this process was found to be process "
                f"{found.rank} of an MPI run of {found.size} processes, and a run uses one of "
                "the two; initialize jax.distributed before the first call of inf.world() or "
                "ds.lower, or do not initialize it."
            )
        return found
    announced = _launched_size()
    comm = None
    if _jax_distributed() or ("mpi4py.MPI" not in sys.modules and announced <= 1):
        found = World(0, 1)
    else:
        try:
            from mpi4py import MPI
        except ImportError as error:
            if announced > 1:
                raise ConfigurationError(
                    f"this process is one of {announced} processes of an MPI launch, and mpi4py "
                    "cannot be imported, so every process would run the whole study on its "
                    "own; install mpi4py built against the MPI library of the launch, or launch "
                    "one process."
                ) from error
            found = World(0, 1)
        else:
            size = int(MPI.COMM_WORLD.Get_size())
            found = World(int(MPI.COMM_WORLD.Get_rank()), size)
            comm = MPI.COMM_WORLD if size > 1 else None
            if (
                size > 1
                and "mpi4py.run" not in sys.modules
                and "mpi4py.__main__" not in sys.modules
            ):
                warnings.warn(
                    "the run has several processes and was not started with python -m mpi4py: an "
                    "exception in one process can then leave the others waiting forever; "
                    "launch with python -m mpi4py script.py.",
                    RuntimeWarning,
                    stacklevel=2,
                )
    _WORLD.append((found, comm))
    return found


def _communicator() -> Any:
    """The MPI communicator of this process's world; ``None`` for a world of one process."""
    world()
    return _WORLD[0][1]


class Transport:
    """The messages between process 0 and the workers of one world.

    Process 0 sends with ``broadcast`` and ``broadcast_array`` and receives
    every worker's reply with ``replies``; a worker receives with ``header``
    and ``array`` and answers with ``reply``.
    """

    def __init__(self, world: World):
        """Wrap the communicator of ``world``, this process's world of more than one process."""
        comm = _communicator()
        if comm is None or tuple(world) != tuple(_WORLD[0][0]):
            raise ValueError("a transport needs this process's world of more than one process.")
        self.world = world
        self.comm = comm
        self.rank = int(world.rank)
        self.size = int(world.size)

    # Process 0 ----------------------------------------------------------------------

    def broadcast(self, header: dict) -> None:
        """Send ``header`` (a small picklable mapping with ``message``) to every worker."""
        self.comm.bcast(header, root=0)

    def broadcast_array(self, array: np.ndarray) -> None:
        """Send a contiguous numpy array whose shape and dtype the header announced."""
        self.comm.Bcast(np.ascontiguousarray(array), root=0)

    def _wait(self, request: Any, deadline: float, message: str, rank: int) -> None:
        while not request.Test():
            if time.monotonic() > deadline:
                who = f"process {rank} sent no reply to" if rank >= 0 else "a process did not reach"
                self.abort(
                    f"{who} {message} within the deadline; the run is aborted and resumes from "
                    "its checkpoint."
                )
            time.sleep(_POLL_SECONDS)

    def _receive(self, source: int, deadline: float, message: str) -> tuple:
        length = np.empty(1, dtype=np.int64)
        self._wait(
            self.comm.Irecv(length, source=source, tag=_TAG_LENGTH), deadline, message, source
        )
        body = np.empty(int(length[0]), dtype=np.uint8)
        self._wait(self.comm.Irecv(body, source=source, tag=_TAG_HEADER), deadline, message, source)
        header = pickle.loads(body.tobytes())
        arrays = []
        for shape, dtype in header.pop("_arrays", ()):
            buffer = np.empty(tuple(shape), dtype=np.dtype(dtype))
            self._wait(
                self.comm.Irecv(buffer, source=source, tag=_TAG_ARRAY), deadline, message, source
            )
            arrays.append(buffer)
        return header, arrays

    def replies(self, message: str, seconds: float) -> list:
        """Every worker's reply to ``message``, in rank order: ``(header, arrays)`` each.

        A worker that has not replied within ``seconds`` aborts the run.
        """
        deadline = time.monotonic() + float(seconds)
        return [self._receive(rank, deadline, message) for rank in range(1, self.size)]

    def acknowledgements(self, message: str, seconds: float) -> None:
        """Wait until every worker has acknowledged ``message``, in rank order.

        A worker acknowledges with a reply whose ``message`` is ``message``.
        Replies to an earlier request that process 0 left without collecting
        them (an exception between its broadcast and its gather) arrive first
        and are dropped. A worker that has not acknowledged within ``seconds``
        aborts the run.
        """
        deadline = time.monotonic() + float(seconds)
        for rank in range(1, self.size):
            header, _ = self._receive(rank, deadline, message)
            while header.get("message") != message:
                header, _ = self._receive(rank, deadline, message)

    def barrier(self, message: str, seconds: float) -> None:
        """Wait for every process at a barrier; process 0 aborts the run after ``seconds``.

        Every process enters the same non-blocking barrier: a blocking and a
        non-blocking collective never match each other.
        """
        request = self.comm.Ibarrier()
        if self.rank == 0:
            self._wait(request, time.monotonic() + float(seconds), message, -1)
        else:
            request.Wait()

    def abort(self, reason: str) -> None:
        """Print ``reason`` and end every process of the run."""
        print(f"rank {self.rank}: {reason}", file=sys.stderr, flush=True)
        self.comm.Abort(_ABORT_CODE)

    # Workers ------------------------------------------------------------------------

    def header(self) -> dict:
        """The next header process 0 broadcasts."""
        return self.comm.bcast(None, root=0)

    def array(self, shape: Any, dtype: Any) -> np.ndarray:
        """The next array process 0 broadcasts, into a fresh buffer."""
        buffer = np.empty(tuple(shape), dtype=np.dtype(dtype))
        self.comm.Bcast(buffer, root=0)
        return buffer

    def reply(self, header: dict, arrays: tuple = ()) -> None:
        """Answer process 0 with ``header`` and the numpy ``arrays`` it announces."""
        arrays = tuple(np.ascontiguousarray(array) for array in arrays)
        header = dict(header)
        header["_arrays"] = tuple((array.shape, array.dtype.str) for array in arrays)
        body = np.frombuffer(pickle.dumps(header, protocol=pickle.HIGHEST_PROTOCOL), np.uint8)
        self.comm.Send(np.asarray([body.shape[0]], dtype=np.int64), dest=0, tag=_TAG_LENGTH)
        self.comm.Send(body, dest=0, tag=_TAG_HEADER)
        for array in arrays:
            self.comm.Send(array, dest=0, tag=_TAG_ARRAY)

    def allgather(self, value: Any) -> list:
        """Every process's ``value``, in rank order, on every process."""
        return self.comm.allgather(value)

    def finalize(self) -> None:
        """Finalize MPI in this process."""
        from mpi4py import MPI

        if not MPI.Is_finalized():
            MPI.Finalize()


__all__ = [
    "DEADLINE_FACTOR",
    "DEADLINE_FLOOR_SECONDS",
    "MESSAGES",
    "PREPARE_DEADLINE_SECONDS",
    "Transport",
    "World",
    "evaluation_deadline",
    "prepare_deadline",
    "reply_deadline",
    "world",
]
