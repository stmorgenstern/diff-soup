"""Compiling the programs of a lowered study, and the report of what was compiled.

[prepare][diff_soup.inference.engine._prepare.prepare] compiles, before any
evaluation, the program of every bucket in every requested mode. A program that
the study holds already, or that the process-wide registry holds for an equal
key, is reused; any other is compiled by
[compile_program][diff_soup.inference.engine._prepare.compile_program]: traced
with the arrays of the bucket's first call, lowered and compiled. Nothing is
solved.

Each program is measured as it is compiled: host wall seconds of tracing,
lowering and compiling; the resident memory of the process before and after,
and its peak; the device memory the compiled program needs, from its memory
analysis; whether JAX's persistent compilation cache served it; and a SHA-256
fingerprint of the lowered text, equal across processes for an equal program.
A program that needs more device memory than the budget (0.8 of the device
allocator's limit) is refused with the numbers that let the caller choose a
narrower ``chunk=``; it is not registered, and nothing is compiled again.

The result is a [PrepareReport][diff_soup.inference.engine._prepare.PrepareReport]:
one [ProgramRecord][diff_soup.inference.engine._prepare.ProgramRecord] per bucket
and mode, and the seconds of every stage of the study's setup, from lowering its
experiments to compiling its last program.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import os
import sys
import time
import types
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec

from diff_soup.inference.engine._batched import (
    BatchedLanes,
    BucketLanes,
    CompiledProgram,
    ProgramKey,
    _first_call,
    _lane_leaves,
)
from diff_soup.inference.engine._batched import prepare as _prepare_programs
from diff_soup.inference.engine._kernel import batched_kernel

BUDGET_SHARE: float = 0.8
"""The share of the device allocator's memory limit one program may need."""

SETUP_STAGES: tuple = ("experiments lowered", "identity and classes", "lane inputs")
"""The stages of lowering a study, in the order they run."""

HIT, MISS, OFF, REUSED = "hit", "miss", "off", "reused"
"""The cache words of a [ProgramRecord][diff_soup.inference.engine._prepare.ProgramRecord]."""

_REQUESTS = "/jax/compilation_cache/compile_requests_use_cache"
_HITS = "/jax/compilation_cache/cache_hits"
_PROGRAM_STAGES = ("trace", "lowering", "compile")


@dataclasses.dataclass(frozen=True)
class HostBytes:
    """The host memory of the process around one compile, in bytes.

    Attributes:
        before: The resident memory before the program was traced
            (``/proc/self/statm``); ``None`` where the system does not report it.
        after: The resident memory after it was compiled; ``None`` where the
            system does not report it.
        peak: The peak resident memory of the process after the compile
            (``ru_maxrss``); ``None`` where the system does not report it.
    """

    before: int | None
    after: int | None
    peak: int | None

    @property
    def change(self) -> int | None:
        """``after - before``: what the compile added to the resident memory; negative when it shrank."""
        if self.before is None or self.after is None:
            return None
        return self.after - self.before


@dataclasses.dataclass(frozen=True)
class ProgramRecord:
    """What ``prepare()`` did for one program: the program of one bucket in one mode.

    Attributes:
        bucket: The index of the bucket in the plan of the study that was
            prepared.
        identity: The first twelve hexadecimal characters of the bucket's
            identity.
        mode: The mode of the lane kernel.
        chunk: ``W``, the units one call of the program evaluates.
        trace_seconds: Host wall seconds of placing the arrays of the bucket's
            first call on the device and tracing the kernel with them;
            ``None`` for a reused program.
        lowering_seconds: Host wall seconds of lowering the trace and taking
            the fingerprint of the lowered text; ``None`` for a reused program.
        compile_seconds: Host wall seconds of compiling the lowered program and
            reading its memory analysis; ``None`` for a reused program.
        host_bytes: The [HostBytes][diff_soup.inference.engine._prepare.HostBytes]
            around the compile; ``None`` for a reused program.
        device_bytes: The device memory the compiled program needs: its
            arguments, outputs, temporaries and code, less the bytes its
            outputs share with its arguments. ``None`` when the backend gives no
            memory analysis.
        cache: ``"hit"`` when JAX's persistent compilation cache served the
            compile, ``"miss"`` when the cache was asked and did not, ``"off"``
            when the cache is disabled or has no directory, ``"reused"`` when
            the program was compiled before this call, for this study or
            another one in the process.
        fingerprint: The SHA-256 of the lowered text, in hexadecimal; equal
            programs have equal fingerprints in every process.
        rank: The process that prepared the program: 0 in a run of one
            process; in a run of several, the rank of the process that holds
            lanes of the bucket. Every such process prepares the bucket's
            programs, and its record names it.
        summed_rows: The names of the bordered rows whose derivative the
            program's Jacobian forms by summing the Jacobian rows of the field
            the row reduces (a ``reduce_residual`` row, or the current row of an
            operational table whose current is a field's residual), instead of
            differentiating the residual densely. Empty when every bordered row
            is differentiated densely.
    """

    bucket: int
    identity: str
    mode: str
    chunk: int
    trace_seconds: float | None
    lowering_seconds: float | None
    compile_seconds: float | None
    host_bytes: HostBytes | None
    device_bytes: int | None
    cache: str
    fingerprint: str
    rank: int = 0
    summed_rows: tuple = ()


@dataclasses.dataclass(frozen=True)
class PrepareReport:
    """What ``prepare()`` compiled, and how long every stage of the setup took.

    Attributes:
        records: One [ProgramRecord][diff_soup.inference.engine._prepare.ProgramRecord]
            per bucket and mode, buckets in the plan's order.
        stages: Host wall seconds by stage: the stages of lowering the study
            (``SETUP_STAGES``), then, per program compiled by this call, its
            trace, lowering and compile, named ``"bucket <index> <mode> <stage>"``.
        seconds: Host wall seconds of this ``prepare()`` call.
        placement: In a run of several processes, the
            [Placement][diff_soup.inference.engine.Placement] of the lanes
            over the processes and the costs it balanced; ``None`` in one
            process. ``records`` then holds one record per process and program
            it prepared, buckets in the plan's order, then modes, then ranks.
    """

    records: tuple
    stages: Mapping
    seconds: float
    placement: Any = None

    @property
    def programs(self) -> int:
        """The number of programs the study holds for the prepared modes."""
        return len(self.records)

    @property
    def compiled(self) -> int:
        """The number of programs this call compiled, served by the persistent cache or not."""
        return sum(record.cache != REUSED for record in self.records)

    @property
    def hits(self) -> int:
        """The number of compiles the persistent compilation cache served."""
        return sum(record.cache == HIT for record in self.records)

    @property
    def misses(self) -> int:
        """The number of compiles that asked the persistent compilation cache and missed."""
        return sum(record.cache == MISS for record in self.records)

    @property
    def reused(self) -> int:
        """The number of programs compiled before this call."""
        return sum(record.cache == REUSED for record in self.records)

    def __str__(self) -> str:
        header = (
            "bucket",
            "identity",
            "mode",
            "W",
            "cache",
            "trace s",
            "lowering s",
            "compile s",
            "device",
            "host change",
            "host peak",
        )
        rows = [header] + [_record_row(record) for record in self.records]
        if self.placement is not None:
            rows = [("rank",) + header] + [
                (str(record.rank),) + row for record, row in zip(self.records, rows[1:])
            ]
            header = rows[0]
        sizes = [max(len(row[column]) for row in rows) for column in range(len(header))]
        lines = [
            "  ".join(cell.ljust(size) for cell, size in zip(row, sizes)).rstrip() for row in rows
        ]
        lines.append("")
        lines.append(
            f"programs {self.programs}: compiled {self.compiled} (persistent cache: "
            f"{self.hits} hits, {self.misses} misses), reused {self.reused}"
        )
        summed = {}
        for record in self.records:
            if record.summed_rows:
                summed.setdefault(record.bucket, record.summed_rows)
        if summed:
            parts = "; ".join(
                f"bucket {bucket}: {', '.join(rows)}" for bucket, rows in sorted(summed.items())
            )
            lines.append(f"bordered rows summed from a field's Jacobian rows: {parts}")
        setup = [(name, self.stages[name]) for name in SETUP_STAGES if name in self.stages]
        if setup:
            parts = ", ".join(f"{name} {seconds:.2f} s" for name, seconds in setup)
            lines.append(f"lowering: {sum(seconds for _, seconds in setup):.2f} s ({parts})")
        lines.append(f"prepare(): {self.seconds:.2f} s")
        if self.placement is not None:
            lines.append(str(self.placement))
        return "\n".join(lines)


def _seconds_cell(seconds: float | None) -> str:
    return "-" if seconds is None else f"{seconds:.2f}"


def _bytes_cell(amount: int | None) -> str:
    if amount is None:
        return "-"
    size, unit = float(abs(amount)), "B"
    for larger in ("KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0:
            break
        size, unit = size / 1024.0, larger
    sign = "-" if amount < 0 else ""
    return f"{sign}{size:.0f} {unit}" if unit == "B" else f"{sign}{size:.1f} {unit}"


def _record_row(record: ProgramRecord) -> tuple:
    host = record.host_bytes
    return (
        str(record.bucket),
        record.identity,
        record.mode,
        str(record.chunk),
        record.cache,
        _seconds_cell(record.trace_seconds),
        _seconds_cell(record.lowering_seconds),
        _seconds_cell(record.compile_seconds),
        "not reported" if record.device_bytes is None else _bytes_cell(record.device_bytes),
        "-" if host is None else _bytes_cell(host.change),
        "-" if host is None else _bytes_cell(host.peak),
    )


def memory_budget(device: Any, share: float | None = None) -> int | None:
    """The device memory, in bytes, one program may need on ``device``.

    Args:
        device: A JAX device.
        share: The share of the allocator's limit one program may take;
            ``None`` takes ``BUDGET_SHARE``.

    Returns:
        ``share`` of the limit the device's allocator reports
        (``memory_stats()["bytes_limit"]``), or ``None`` when the device
        reports no limit, as a CPU does; then no program is refused for its
        memory.
    """
    stats = device.memory_stats()
    if not stats or "bytes_limit" not in stats:
        return None
    share = BUDGET_SHARE if share is None else float(share)
    return int(share * int(stats["bytes_limit"]))


@contextlib.contextmanager
def cache_writes(enabled: bool) -> Iterator[None]:
    """Let compiles write JAX's persistent compilation cache, or keep them from writing it.

    With ``enabled=False`` the minimum compile time an entry needs to be
    written is raised to infinity for the block, so nothing is written while
    every entry already in the cache is still read. A run of several
    processes compiles every program with writes on in one process only, then
    with writes off in the others, since the cache has no safe path for
    several writers of one entry.
    """
    if enabled:
        yield
        return
    name = "jax_persistent_cache_min_compile_time_secs"
    before = getattr(jax.config, name)
    jax.config.update(name, float("inf"))
    try:
        yield
    finally:
        jax.config.update(name, before)


def _resident_bytes() -> int | None:
    """The resident memory of this process, from ``/proc/self/statm``."""
    try:
        with open("/proc/self/statm") as statm:
            pages = int(statm.read().split()[1])
    except OSError:
        return None
    return pages * os.sysconf("SC_PAGE_SIZE")


def _peak_bytes() -> int | None:
    """The peak resident memory of this process, from ``getrusage``."""
    try:
        import resource
    except ImportError:
        return None
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports kibibytes; macOS reports bytes.
    return peak if sys.platform == "darwin" else peak * 1024


def _device_bytes(compiled: Any) -> int | None:
    """Arguments + outputs + temporaries + code - aliases of a compiled program."""
    analysis = compiled.memory_analysis()
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
        "alias_size_in_bytes",
    )
    sizes = [getattr(analysis, name, None) for name in names]
    if analysis is None or any(size is None for size in sizes):
        return None
    arguments, outputs, temporaries, code, aliases = (int(size) for size in sizes)
    return arguments + outputs + temporaries + code - aliases


class _CacheRequests:
    """Counts the persistent-cache requests and hits JAX reports."""

    def __init__(self) -> None:
        self.requests = 0
        self.hits = 0

    def __call__(self, event: str, **kwargs: Any) -> None:
        if event == _REQUESTS:
            self.requests += 1
        elif event == _HITS:
            self.hits += 1

    @property
    def word(self) -> str:
        # JAX asks its cache even when no cache directory is set, so nothing can be stored
        # or loaded; that is reported as off.
        if self.hits:
            return HIT
        return MISS if self.requests and jax.config.jax_compilation_cache_dir else OFF


@contextlib.contextmanager
def _cache_requests() -> Iterator[_CacheRequests]:
    listener = _CacheRequests()
    jax.monitoring.register_event_listener(listener)
    try:
        yield listener
    finally:
        jax.monitoring.unregister_event_listener(listener)


def _largest_fitting_width(width: int, needs: int, budget: int) -> int:
    """The largest power of two ``p`` with ``p * needs / width`` within ``budget``; 0 if none."""
    units = (budget * width) // needs
    return 0 if units < 1 else 1 << (units.bit_length() - 1)


class ProgramOverBudget(ValueError):
    """A compiled program needs more device memory than one program may take.

    Attributes:
        bucket: The bucket's index in the plan that was prepared.
        identity: The first twelve hexadecimal characters of its identity.
        mode: The mode of the program.
        chunk: ``W``, its chunk width.
        needs: The device memory it needs, in bytes.
        budget: The budget it exceeds, in bytes.
        fits: The largest power-of-two chunk width that fits; 0 when none.
    """

    def __init__(self, message: str, **numbers: Any):
        """Carry the message and the numbers that make it."""
        super().__init__(message)
        for name, value in numbers.items():
            setattr(self, name, value)


def _over_budget(
    entry: BucketLanes, key: ProgramKey, needs: int, budget: int, device: Any
) -> ValueError:
    bucket = entry.bucket
    fits = _largest_fitting_width(key.chunk, needs, budget)
    if fits:
        remedy = (
            f"the largest power-of-two width that fits is {fits}: lower the study with "
            f"ds.lower(study, chunk={fits}), or, for a study cut for a population of points, "
            f"take LoweredStudy.for_population(members, chunk={fits}); any chunk= whose program "
            "fits the budget will do"
        )
    else:
        remedy = (
            "a program of one unit per call does not fit either, so no chunk= of "
            "ds.lower(study, chunk=) or LoweredStudy.for_population(members, chunk=) makes it "
            "fit on this device"
        )
    return ProgramOverBudget(
        f"the {key.mode} program of bucket {bucket.index} (identity {bucket.identity}) at the "
        f"chunk width W = {key.chunk} needs {needs} bytes of device memory ({_bytes_cell(needs)}), "
        f"more than the budget of {budget} bytes ({_bytes_cell(budget)}) one program may take on "
        f"{device}; {remedy}.",
        bucket=int(bucket.index),
        identity=str(bucket.identity),
        mode=str(key.mode),
        chunk=int(key.chunk),
        needs=int(needs),
        budget=int(budget),
        fits=int(fits),
    )


def _summed_rows(program: Any) -> tuple:
    """The bordered rows whose derivative the program's Jacobian sums from a field's rows."""
    view = program.problem
    source = getattr(view, "source", None)
    kernel = getattr(view if source is None else source, "_physics_jac_blocks_fn", None)
    plan = getattr(kernel, "constraint_border_plan", None)
    if plan is None:
        return ()
    return tuple(str(name) for name in plan.shortcut_constraints)


def split_empty_leaves(inputs: Any) -> tuple[list, Callable[[list, int], Any]]:
    """Keep the zero-size leaves of lane-axis inputs out of a ``shard_map``.

    XLA's Shardy partitioner (jaxlib 0.11) crashes the process with a
    segmentation fault while compiling a ``shard_map`` whose body reads a
    zero-size operand. Lane inputs hold such leaves whenever a per-lane input
    has no entries, for example the frozen parameters of lanes that freeze
    none. A zero-size leaf carries no data, so it stays outside the shard
    boundary and the body rebuilds it as zeros.

    Args:
        inputs: A pytree whose every leaf leads with the lane axis.

    Returns:
        ``(kept, restore)``: the leaves that hold data, as a list to pass
        through the shard boundary, and ``restore(kept_local, n_local)``, which
        rebuilds the full pytree inside the body from the local kept leaves
        and the local lane count.
    """
    leaves, treedef = jax.tree_util.tree_flatten(inputs)
    empty = [int(np.prod(jnp.shape(leaf))) == 0 for leaf in leaves]
    shapes = [(tuple(jnp.shape(leaf)[1:]), jnp.result_type(leaf)) for leaf in leaves]
    kept = [leaf for leaf, is_empty in zip(leaves, empty) if not is_empty]

    def restore(kept_local: list, n_local: int) -> Any:
        local = iter(kept_local)
        rebuilt = [
            jnp.zeros((n_local, *shape), dtype) if is_empty else next(local)
            for is_empty, (shape, dtype) in zip(empty, shapes)
        ]
        return jax.tree_util.tree_unflatten(treedef, rebuilt)

    return kept, restore


def placed_kernel(program: Any, mode: str, mesh: Any = None) -> Any:
    """The kernel of one call of a bucket's program, placed as the plan places the lanes.

    Args:
        program: The bucket's [LaneProgram][diff_soup.inference.engine._kernel.LaneProgram].
        mode: The mode of the lane kernel.
        mesh: The mesh whose data axis every call is split over, or ``None``.

    Returns:
        [batched_kernel][diff_soup.inference.engine._kernel.batched_kernel]
        without a mesh. On a mesh, that kernel mapped over the devices of its
        ``data`` axis with ``jax.shard_map``: every argument and result splits
        its leading lane axis over those devices, each device runs the batched
        kernel on its share of the call's units, and nothing is communicated
        between them. One program serves every device. Zero-size argument
        leaves stay outside the mapped body
        ([split_empty_leaves][diff_soup.inference.engine._prepare.split_empty_leaves]).
    """
    kernel = batched_kernel(program, mode)
    if mesh is None:
        return kernel
    split = PartitionSpec("data")

    def placed(points: Any, lanes: Any) -> Any:
        kept, restore = split_empty_leaves((points, lanes))

        def body(*local: Any) -> Any:
            return kernel(*restore(list(local), local[0].shape[0]))

        return jax.shard_map(body, mesh=mesh, in_specs=split, out_specs=split, check_vma=False)(
            *kept
        )

    return jax.jit(placed)


def compile_program(
    lanes: BatchedLanes, entry: BucketLanes, key: ProgramKey, *, budget_share: float | None = None
) -> CompiledProgram:
    """Trace, lower and compile the program of ``key`` with the bucket's first call, measured.

    Args:
        lanes: The lanes of the study.
        entry: The bucket whose program is compiled.
        key: The program's [ProgramKey][diff_soup.inference.engine._batched.ProgramKey].
        budget_share: The share of the device allocator's limit the program
            may need; ``None`` takes ``BUDGET_SHARE``.

    Returns:
        The [CompiledProgram][diff_soup.inference.engine._batched.CompiledProgram],
        its record carrying the measurements of this compile.

    Raises:
        ValueError: If the compiled program needs more device memory than
            [memory_budget][diff_soup.inference.engine._prepare.memory_budget]
            allows; the message names the bucket, the mode, the width, the
            memory needed, the budget and the largest power-of-two width that
            fits.
    """
    resident = _resident_bytes()
    with _cache_requests() as cache:
        start = time.perf_counter()
        points, inputs = _first_call(lanes, entry, key.chunk)
        kernel = placed_kernel(entry.program, key.mode, lanes.plan.mesh)
        traced = kernel.trace(points, inputs)
        traced_at = time.perf_counter()
        lowered = traced.lower()
        fingerprint = hashlib.sha256(lowered.as_text().encode("utf-8")).hexdigest()
        lowered_at = time.perf_counter()
        compiled = lowered.compile()
        needs = _device_bytes(compiled)
        compiled_at = time.perf_counter()
    device = lanes.plan.device
    budget = memory_budget(device, budget_share)
    if budget is not None and needs is not None and needs > budget:
        raise _over_budget(entry, key, needs, budget, device)
    record = ProgramRecord(
        bucket=int(entry.bucket.index),
        identity=str(entry.bucket.identity),
        mode=str(key.mode),
        chunk=int(key.chunk),
        trace_seconds=traced_at - start,
        lowering_seconds=lowered_at - traced_at,
        compile_seconds=compiled_at - lowered_at,
        host_bytes=HostBytes(before=resident, after=_resident_bytes(), peak=_peak_bytes()),
        device_bytes=needs,
        cache=cache.word,
        fingerprint=fingerprint,
        summed_rows=_summed_rows(entry.program),
    )
    return CompiledProgram(
        key=key,
        program=entry.program,
        kernel=kernel,
        compiled=compiled,
        lanes_tree=entry.tree,
        lane_leaves=_lane_leaves(entry),
        record=record,
    )


def default_modes(lanes: BatchedLanes) -> tuple:
    """The modes ``prepare()`` compiles when none are named.

    ``value`` and ``blocks`` when every bucket's solver differentiates forward;
    otherwise ``value`` and ``gradient``, which every bucket runs, since a
    solver with a reverse adjoint has no ``blocks`` mode.
    """
    if all("blocks" in entry.bucket.chunk for entry in lanes.buckets):
        return ("value", "blocks")
    return ("value", "gradient")


def prepare(lanes: BatchedLanes, *, modes: Any = None, setup: Mapping) -> PrepareReport:
    """Compile the program of every bucket of a study in every mode of ``modes``.

    Args:
        lanes: The lanes of the study.
        modes: The modes to compile; ``None`` takes
            [default_modes][diff_soup.inference.engine._prepare.default_modes].
        setup: The seconds of the stages of lowering the study, by name.

    Returns:
        The [PrepareReport][diff_soup.inference.engine._prepare.PrepareReport].

    Raises:
        ValueError: If a mode is not a mode of the lane kernel, a bucket's
            solver cannot run a mode, or a program needs more device memory
            than the budget.
    """
    modes = default_modes(lanes) if modes is None else modes
    start = time.perf_counter()
    prepared = _prepare_programs(lanes, modes=modes, compile_program=compile_program)
    seconds = time.perf_counter() - start
    records, stages = [], dict(setup)
    for entry in prepared:
        compiled = entry.program.record
        if entry.reused:
            records.append(
                dataclasses.replace(
                    compiled,
                    bucket=int(entry.bucket),
                    trace_seconds=None,
                    lowering_seconds=None,
                    compile_seconds=None,
                    host_bytes=None,
                    cache=REUSED,
                )
            )
            continue
        records.append(compiled)
        spent = (compiled.trace_seconds, compiled.lowering_seconds, compiled.compile_seconds)
        for stage, stage_seconds in zip(_PROGRAM_STAGES, spent):
            stages[f"bucket {entry.bucket} {entry.mode} {stage}"] = stage_seconds
    return PrepareReport(
        records=tuple(records),
        stages=types.MappingProxyType(stages),
        seconds=seconds,
    )


__all__ = [
    "BUDGET_SHARE",
    "SETUP_STAGES",
    "HostBytes",
    "PrepareReport",
    "ProgramRecord",
    "compile_program",
    "default_modes",
    "memory_budget",
    "placed_kernel",
    "prepare",
    "split_empty_leaves",
]
