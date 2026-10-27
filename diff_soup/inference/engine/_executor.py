"""The lanes of one lowered study evaluated by several processes: the coordinator and the workers.

Every process of the run lowers the same study and builds the same plan
(``PLAN_CHECK`` refuses a mismatch). Process 0 places the lanes
([place][diff_soup.inference.engine._placement.place]) and every process keeps
its own lanes and builds their inputs on its one device (``PLACE``). Process 0
then holds a [Coordinator][diff_soup.inference.engine._executor.Coordinator]:
the lowered study's ``prepare`` and ``evaluate`` go through it, and it evaluates
its own lanes like any worker. Every other process enters
[Worker.serve][diff_soup.inference.engine._executor.Worker.serve] inside
``ds.lower`` and never returns from it: it answers ``PREPARE`` and ``EVALUATE``
until ``SHUTDOWN``, then frees its device memory, runs its exit handlers,
acknowledges, finalizes MPI and raises ``SystemExit(0)``; process 0's
``close()`` returns once every worker has acknowledged. A script written as
``with ds.distributed.run(study) as lowered:``
([run][diff_soup._distributed.run]) states this contract: the block runs on
process 0, and the block's end is the workers' exit.

An evaluation sends the request's header and points to every process. Each
process evaluates its own units through its compiled programs, sums its lanes
over the study's level tables (every lane weighing one) into one partial table
per member, and replies with those sums and its lanes' costs, rows, statuses,
work and seconds. Process 0 adds the partial tables in rank order and returns
[Results][diff_soup.inference.engine.Results] whose ``levels`` carry the sums
and whose ``normal`` and ``gradient`` are ``None``: the lanes' own matrices
never leave the process that computed them. A request of many members is sent
in groups of members whose sums fit the wire budget.

What a series of evaluations learns (the step memory that orders a call's
units, the inputs kept on the device) belongs to the lanes it went through on
process 0: the lowered study's own, or a fit's own copy (every fit of a
lowered study evaluates through lanes of its own). Every process mirrors it:
the header of an evaluation carries the fit it belongs to, ``None`` for the
study's own lanes, and each process evaluates through its lanes of that fit,
which share the study's programs and inputs on the host and start their own
state empty. A process keeps the study's state and the state of the fit it
evaluated last; the first evaluation of another fit releases the one before,
and the end of the run releases both.

A prepare compiles every process's programs for the requested modes in two
phases: first each program by the lowest process that holds its bucket, with
writes to JAX's persistent compilation cache on, then, after a barrier, every
other process's copies with writes off, so that one process writes each cache
entry and the others read it. The report gathers one record per process and
program.
"""

from __future__ import annotations

import atexit
import dataclasses
import gc
import hashlib
import sys
import time
import types
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np

from diff_soup._errors import BackendError
from diff_soup.inference.engine import _batched
from diff_soup.inference.engine import _prepare as preparing
from diff_soup.inference.engine import _transport as transports
from diff_soup.inference.engine._ledger import attributed_seconds
from diff_soup.inference.engine._levels import LevelBlocks, LevelGradients, partial_levels
from diff_soup.inference.engine._order import canonical_lanes
from diff_soup.inference.engine._plan import population_plan
from diff_soup.inference.engine._request import Request
from diff_soup.inference.engine._result import Batch, Results, Status, Work
from diff_soup.inference.engine._status import ABSENT
from diff_soup.inference.engine._transport import Transport

WIRE_BUDGET: int = 1 << 30
"""The bytes of partial tables process 0 receives for one group of members, at most."""

_REDUCED = ("blocks", "hessian", "gradient")
_WORK = ("accepts", "rejects", "trial_steps", "newton_iters", "jac_refreshes", "factorizations",
         "first_step")  # fmt: skip


class Remote(NamedTuple):
    """A program the processes of a run hold, recorded where process 0 keeps its lanes' programs.

    Attributes:
        key: The [ProgramKey][diff_soup.inference.engine._batched.ProgramKey].
    """

    key: Any


def plan_fingerprint(plan: Any) -> str:
    """The SHA-256 of what every process must agree on: lanes, bucket identities, widths, chunks."""
    digest = hashlib.sha256()
    for bucket in plan.buckets:
        identity = getattr(bucket.full_identity, "digest", None) or str(bucket.full_identity)
        chunks = sorted((mode, int(entry.width)) for mode, entry in bucket.chunk.items())
        digest.update(
            repr(
                (
                    int(bucket.index),
                    str(identity),
                    tuple(str(lane) for lane in bucket.lanes),
                    tuple(int(child) for child in bucket.children),
                    int(bucket.width),
                    chunks,
                )
            ).encode("utf-8")
        )
    digest.update(repr(tuple(str(entry.lane) for entry in plan.refused)).encode("utf-8"))
    return digest.hexdigest()


@dataclasses.dataclass(eq=False)
class Shard:
    """One process's lanes of one cut of the study.

    Attributes:
        lanes: The [BatchedLanes][diff_soup.inference.engine._batched.BatchedLanes]
            of the process's lanes, with their inputs and programs.
        origin: Per bucket of ``lanes``, its index in the study's plan.
    """

    lanes: Any
    origin: tuple


def _pack_levels(sums: Any) -> tuple:
    """``(kind, leaves)`` of level sums whose leaves lead with the member axis."""
    depth = len(sums.cost) - 1
    leaves = [np.asarray(sums.gradient[level]) for level in range(1, depth + 1)]
    leaves += [np.asarray(sums.cost[level]) for level in range(1, depth + 1)]
    leaves += [np.asarray(sums.rows[level]) for level in range(1, depth + 1)]
    leaves.append(np.asarray(sums.gradient_shared))
    if isinstance(sums, LevelBlocks):
        leaves += [np.asarray(sums.diagonal[level]) for level in range(1, depth + 1)]
        leaves += [
            np.asarray(entry) for level in range(1, depth + 1) for entry in sums.cross[level]
        ]
        leaves.append(np.asarray(sums.shared))
        return "blocks", leaves
    return "gradient", leaves


def _unpack_levels(kind: str, depth: int, leaves: list) -> Any:
    """The level sums ``_pack_levels`` packed."""
    leaves = list(leaves)

    def take(count: int) -> tuple:
        taken = tuple(leaves[:count])
        del leaves[:count]
        return taken

    gradient = (None,) + take(depth)
    cost = (None,) + take(depth)
    rows = (None,) + take(depth)
    (gradient_shared,) = take(1)
    if kind == "gradient":
        return LevelGradients(
            gradient=gradient, gradient_shared=gradient_shared, cost=cost, rows=rows
        )
    diagonal = (None,) + take(depth)
    cross = [None]
    for level in range(1, depth + 1):
        cross.append(take(level))
    (shared,) = take(1)
    return LevelBlocks(
        diagonal=diagonal,
        cross=tuple(cross),
        gradient=gradient,
        cost=cost,
        rows=rows,
        shared=shared,
        gradient_shared=gradient_shared,
    )


def _numbers_per_member(tables: Any, mode: str) -> int:
    """The numbers of one member's level sums in ``mode``."""
    widths, counts = tables.widths, tables.counts
    size = 2 * sum(counts) + int(counts[0]) * int(widths[0])
    size += sum(int(count) * int(width) for count, width in zip(counts, widths[1:]))
    if mode != "gradient":
        size += int(counts[0]) * int(widths[0]) ** 2
        for level in range(1, len(widths)):
            size += (
                int(counts[level - 1])
                * int(widths[level])
                * (int(widths[level]) + sum(int(width) for width in widths[:level]))
            )
    return int(size)


class Cluster:
    """What every process of a run holds: the transport, the placement, its own lanes, the tables.

    Attributes:
        transport: The [Transport][diff_soup.inference.engine._transport.Transport].
        study: The lowered study of this process (on a worker, the one under
            construction, which never returns).
        metadata: The study's lanes without inputs, for its own plan.
        placement: The [Placement][diff_soup.inference.engine._placement.Placement].
        fingerprint: The plan's fingerprint every process agreed on.
        shards: Per cut (``None``, or ``(members, chunk)``), this process's
            lanes of the study, with the study's own evaluation state.
        fit: The fit whose state ``fit_shards`` hold; ``None`` before any.
        fit_shards: Per cut, this process's lanes of ``fit``: the study's
            lanes with a step memory and device inputs of their own.

    The ledger of the lanes' evaluations is process 0's alone: every
    evaluation is requested and recorded there, and a worker records nothing.
    """

    def __init__(self, transport: Transport, study: Any, joint: Any, metadata: Any, placement: Any):
        """Keep this process's lanes of the study's plan and build their inputs."""
        self.transport = transport
        self.rank = transport.rank
        self.study = study
        self.joint = joint
        self.metadata = metadata
        self.placement = placement
        self.fingerprint = plan_fingerprint(metadata.plan)
        self.mine = frozenset(placement.lanes_of(self.rank))
        self.shards: dict = {}
        self.fit: Any = None
        self.fit_shards: dict = {}
        self._tables: dict = {}
        self._view: Any = None
        base, origin = _batched.restricted_plan(metadata.plan, self.mine)
        self.shards[None] = Shard(
            lanes=_batched.batched_lanes(base, joint) if base.buckets else None, origin=origin
        )
        self.register(self.shards[None])

    def register(self, shard: Shard) -> None:
        """Register in this process what its lanes need before their first solve.

        The lanes' inputs are all a lane needs today; a sparse solver whose
        pattern is registered once per process registers this process's
        meshes here, also when every program comes from the persistent cache.
        """

    def shard(self, cut: Any, fit: Any = None) -> Shard:
        """This process's lanes of the cut ``cut`` (``None``, or ``(members, chunk)``).

        With ``fit`` (the token process 0 gives a fit), the lanes of that fit:
        the study's lanes of the cut with their own step memory and device
        inputs, sharing the study's programs. The first request of another
        fit releases the state of the one before.
        """
        if fit is None:
            return self._study_shard(cut)
        if fit != self.fit:
            self.fit = fit
            self.fit_shards.clear()
        shard = self.fit_shards.get(cut)
        if shard is None:
            study = self._study_shard(cut)
            lanes = None if study.lanes is None else _batched.with_own_state(study.lanes)
            shard = Shard(lanes=lanes, origin=study.origin)
            self.fit_shards[cut] = shard
        return shard

    def _study_shard(self, cut: Any) -> Shard:
        shard = self.shards.get(cut)
        if shard is not None:
            return shard
        base = self.shards[None]
        members, chunk = cut
        full = population_plan(self.metadata.plan, members, chunk=chunk)
        restricted, origin = _batched.restricted_plan(full, self.mine)
        lanes = None if base.lanes is None else _batched.replanned_lanes(base.lanes, restricted)
        shard = Shard(lanes=lanes, origin=origin)
        self.shards[cut] = shard
        return shard

    def tables(self, lane_ids: tuple) -> Any:
        """The study's level tables for the lanes ``lane_ids`` (a subset, in canonical order)."""
        key = tuple(lane_ids)
        tables = self._tables.get(key)
        if tables is None:
            from diff_soup.inference.hierarchical.level_solvers import _route_tables

            if self._view is None:
                self._view = self.study.joint_view()
            tables, _ = _route_tables(
                self.study, self._view, key, self.metadata.plan, self.metadata.plan.device
            )
            self._tables[key] = tables
        return tables

    # One process's share of a request ------------------------------------------------

    def local_evaluate(self, header: Mapping, points: np.ndarray) -> tuple:
        """This process's reply to ``EVALUATE``: ``(header, arrays)``.

        Errors of the request or of a program are part of the reply, so that
        every process answers and process 0 raises them.
        """
        try:
            return self._local_evaluate(header, points)
        except (BackendError, ValueError) as error:
            return {"error": (type(error).__name__, str(error)), "lane_ids": ()}, ()

    def _local_evaluate(self, header: Mapping, points: np.ndarray) -> tuple:
        shard = self.shard(header["cut"], header.get("fit"))
        chosen = tuple(header["chosen"])
        columns = [c for c, lane in enumerate(chosen) if lane in self.mine]
        units = header["units"]
        empty = {"lane_ids": ()}
        if shard.lanes is None or not columns:
            return empty, ()
        if units is not None:
            units = np.asarray(units)[:, columns]
            if not np.any(units):
                return empty, ()

        def part(array: Any) -> Any:
            return None if array is None else np.asarray(array)[:, columns]

        own = tuple(chosen[c] for c in columns)
        held = set(own)
        order = tuple(lane for lane in (header["order"] or ()) if lane in held) or None
        request = Request(
            mode=header["mode"],
            points=points,
            lanes=own,
            order=order,
            rungs=part(header["rungs"]),
            first_steps=part(header["first_steps"]),
            units=units,
        )
        kinds, factors = header["retry"]
        results = _batched.evaluate(
            shard.lanes, request, retry_kinds=kinds, retry_first_step_factors=factors
        )
        origin = shard.origin
        reply = {
            "lane_ids": tuple(results.lane_ids),
            "bucket": tuple(int(origin[int(b)]) for b in results.bucket),
            "kind": np.asarray(results.status.kind),
            "t_reached": np.asarray(results.status.t_reached),
            "rung": np.asarray(results.status.rung),
            "cost": np.asarray(results.cost),
            "rows": np.asarray(results.rows),
            "work": {name: np.asarray(getattr(results.work, name)) for name in _WORK},
            "seconds": {int(origin[int(b)]): float(s) for b, s in results.seconds.items()},
            "batches": {int(origin[int(b)]): tuple(batch) for b, batch in results.batches.items()},
            "retries": tuple(results.retries),
            "lane_seconds": attributed_seconds(results),
        }
        arrays: tuple = ()
        if header["mode"] in _REDUCED:
            sums = partial_levels(
                self.tables(tuple(results.lane_ids)), results, self.metadata.plan.device
            )
            kind, leaves = _pack_levels(sums)
            reply["levels"] = (kind, tuple((leaf.shape, leaf.dtype.str) for leaf in leaves))
            flat = np.concatenate([leaf.astype(np.float64).reshape(-1) for leaf in leaves])
            arrays = (flat,)
            if results.normal is not None:
                # Each lane's diagonal, read on the host after the lanes ran.
                reply["normal_diagonal"] = np.array(
                    np.diagonal(results.normal, axis1=-2, axis2=-1), dtype=np.float64
                )
        return reply, arrays

    def local_prepare(self, header: Mapping, phase: str) -> tuple:
        """This process's programs of one phase: ``(records, refusals)``.

        Phase ``"A"`` compiles the programs of the buckets this process is the
        compiling process of, with writes to the persistent cache on; phase
        ``"B"`` the others, with writes off. A program over the memory budget is
        refused, the others still compiled.
        """
        shard = self.shard(header["cut"])
        if shard.lanes is None:
            return (), ()
        assignment = header["assignment"]
        # Every process applies its own share of its own device's memory (BUDGET_SHARE).
        share = None
        records, refusals = [], []

        def compile_with_share(lanes: Any, entry: Any, key: Any) -> Any:
            return preparing.compile_program(lanes, entry, key, budget_share=share)

        writes = phase == "A"
        with preparing.cache_writes(writes):
            for entry in shard.lanes.buckets:
                local = int(entry.bucket.index)
                index = int(shard.origin[local])
                if (int(assignment[index]) == self.rank) != writes:
                    continue
                for mode in header["modes"]:
                    try:
                        prepared = _batched.prepare(
                            shard.lanes,
                            modes=(mode,),
                            compile_program=compile_with_share,
                            buckets=(local,),
                        )
                    except preparing.ProgramOverBudget as refusal:
                        refusals.append(
                            {
                                "rank": self.rank,
                                "bucket": index,
                                "identity": refusal.identity,
                                "mode": refusal.mode,
                                "chunk": refusal.chunk,
                                "needs": refusal.needs,
                                "budget": refusal.budget,
                                "fits": refusal.fits,
                            }
                        )
                        continue
                    for item in prepared:
                        record = item.program.record
                        if item.reused:
                            record = dataclasses.replace(
                                record,
                                trace_seconds=None,
                                lowering_seconds=None,
                                compile_seconds=None,
                                host_bytes=None,
                                cache=preparing.REUSED,
                            )
                        records.append(dataclasses.replace(record, bucket=index, rank=self.rank))
        return tuple(records), tuple(refusals)

    def release(self) -> None:
        """Drop this process's lanes and programs, and the device memory they hold."""
        self.shards.clear()
        self.fit_shards.clear()
        self.fit = None
        self._tables.clear()
        import jax

        jax.clear_caches()
        gc.collect()


def _run_exit_handlers() -> None:
    """Run this process's exit handlers now, in the order Python runs them at exit, and flush.

    The handlers are the functions registered with ``atexit`` (a script's
    record of the process's exit, logging's flush, JAX's own clean-up); each
    runs once, so none runs again when the process exits.
    """
    # No public API runs the registered handlers before the interpreter exits, and they
    # have to run before the acknowledgement that lets process 0 go on.
    atexit._run_exitfuncs()
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and not getattr(stream, "closed", False):
            stream.flush()


class Worker(Cluster):
    """A process above 0: it answers process 0's messages until ``SHUTDOWN``."""

    def serve(self) -> None:
        """Answer ``PREPARE``, ``EVALUATE`` and the private-tier messages; raise ``SystemExit(0)`` on ``SHUTDOWN``.

        Never returns. On ``SHUTDOWN`` the process frees its device memory,
        runs its exit handlers (the functions registered with ``atexit``,
        such as a script's record of the process's exit) and flushes its
        output, acknowledges, finalizes MPI and raises ``SystemExit(0)``.
        The exit handlers run before the acknowledgement because process 0's
        ``close()`` returns only after every acknowledgement: an error that
        aborts the run after ``close()`` then comes after every worker's
        records are written; a worker still waiting in MPI's finalize for
        process 0 ends with the abort's status.

        The ``SystemExit`` unwinds the frames of the script that called
        ``ds.lower``: their ``finally`` clauses run, an ``except Exception``
        does not catch it, and nothing registered with ``atexit`` runs again.
        ``with ds.distributed.run(study) as lowered:``
        ([run][diff_soup._distributed.run]) is the form of a script that runs
        its block on process 0 only. An exception other than a refusal of the
        request ends the process, which ends the run.

        Raises:
            SystemExit: With status 0, once ``SHUTDOWN`` is acknowledged.
        """
        transport = self.transport
        while True:
            header = transport.header()
            message = header["message"]
            if message == "EVALUATE":
                points = transport.array(header["points_shape"], header["points_dtype"])
                reply, arrays = self.evaluate(header, points)
                transport.reply(reply, arrays)
            elif message == "PREPARE":
                first = self.local_prepare(header, "A")
                transport.barrier("PREPARE", transports.prepare_deadline())
                second = self.local_prepare(header, "B")
                transport.reply({"records": first[0] + second[0], "refusals": first[1] + second[1]})
            elif message in ("PRIVATE_SAVE", "PRIVATE_LOAD"):
                # The lanes hold no private coordinates yet: nothing to write or read.
                transport.reply({"message": message})
            elif message == "SHUTDOWN":
                self.release()
                _run_exit_handlers()
                transport.reply({"message": message})
                transport.finalize()
                raise SystemExit(0)
            else:
                raise RuntimeError(f"process {self.rank} received the unknown message {message!r}.")

    def evaluate(self, header: Mapping, points: np.ndarray) -> tuple:
        """This process's reply to ``EVALUATE``."""
        return self.local_evaluate(header, points)


class Coordinator(Cluster):
    """Process 0: the lowered study's evaluations and prepares, sent to every process."""

    def __init__(self, *args: Any, **kwargs: Any):
        """As [Cluster][diff_soup.inference.engine._executor.Cluster]; the run is open."""
        super().__init__(*args, **kwargs)
        self.closed = False
        self._fits = 0
        self._fit_memory: Any = None

    def _fit_token(self, lanes: Any) -> Any:
        """The token of the fit whose lanes ``lanes`` are; ``None`` for the study's own lanes.

        A fit is told apart by its step memory, which every fit's lanes own
        (``with_own_state``); a new memory is a new fit.
        """
        own = self.study.__dict__.get("_lanes")
        if own is None or lanes.memory is own.memory:
            return None
        if lanes.memory is not self._fit_memory:
            # Holding the memory keeps its identity from being reused by a later fit.
            self._fit_memory = lanes.memory
            self._fits += 1
        return self._fits

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError(
                "the processes of this run were shut down (LoweredStudy.close() or the end of "
                "the script); a run of several processes evaluates until its study is closed."
            )

    # Evaluation -----------------------------------------------------------------------

    def evaluate(
        self, lanes: Any, request: Request, *, cut: Any, retry: tuple, longest: float
    ) -> Results:
        """Evaluate ``request`` over every process; the results in canonical order.

        Args:
            lanes: The study's lanes without inputs, for the plan of ``cut``,
                or a fit's copy of them: every process then evaluates through
                its lanes of that fit, with the fit's own evaluation state.
            request: The [Request][diff_soup.inference.engine.Request].
            cut: ``None``, or ``(members, chunk)`` of a study cut for a
                population.
            retry: ``(retry_kinds, retry_first_step_factors)``.
            longest: The longest seconds one evaluation of a lane took so far.

        Returns:
            The [Results][diff_soup.inference.engine.Results]: ``levels`` holds
            every member's sums over the level tables in the ``blocks``,
            ``hessian`` and ``gradient`` modes; ``normal``, ``gradient`` and
            ``observables`` are ``None``.

        Raises:
            ValueError: As a lowered study's ``evaluate`` in one process, and
                for the ``observe`` mode, whose predictions stay in the
                processes that hold the lanes.
            BackendError: If a program raised in a process; the error names
                the process.
        """
        self._check_open()
        if not isinstance(request, Request):
            raise ValueError(f"evaluate takes a Request; got {type(request).__name__}.")
        mode = request.mode
        if mode == "observe":
            raise ValueError(
                "the observe mode returns every lane's predictions at its frames, and the lanes "
                "of this study run in several processes, where their data and predictions stay; "
                "evaluate the observe mode in a run of one process."
            )
        points = _batched._points(lanes, request)
        chosen = canonical_lanes(
            request.lanes,
            request.order,
            study_ids=lanes.lane_ids,
            planned=lanes.where,
            refused=lanes.refused,
        )
        members = request.members
        if not chosen or not members.shape[0]:
            raise ValueError(
                f"the request selects no unit: {len(chosen)} lane(s) and {members.shape[0]} "
                "active member(s); select at least one lane and one member."
            )
        present = request.unit_mask(len(chosen))
        if present is not None and not np.any(present):
            raise ValueError(
                "the request selects no unit: units= leaves out every unit of its members and "
                "lanes; set at least one entry of units= to True."
            )
        bucket_of = np.asarray([lanes.where[lane][0] for lane in chosen], dtype=np.int64)
        touched = sorted(
            set((bucket_of if present is None else bucket_of[np.any(present, axis=0)]).tolist())
        )
        _batched._check_programs(lanes, mode, touched)
        shape = (int(members.shape[0]), len(chosen))
        factors = np.asarray(tuple(float(f) for f in retry[1]), dtype=np.float64)
        rungs, bases = _batched._requested_rungs(request, shape, factors)
        rungs = None if request.rungs is None else rungs
        bases = None if request.first_steps is None else bases
        fit = self._fit_token(lanes)

        tables = None
        size = 0
        if mode in _REDUCED:
            tables = self.study._route().tables
            size = _numbers_per_member(tables, mode)
        group = shape[0]
        if size:
            group = max(1, min(shape[0], WIRE_BUDGET // max(8 * size * self.transport.size, 1)))
        parts = []
        for first in range(0, shape[0], group):
            rows = np.arange(first, min(first + group, shape[0]))
            replies = self._evaluate_rows(
                fit,
                mode,
                points[rows],
                chosen,
                request,
                cut,
                retry,
                longest,
                None if rungs is None else rungs[rows],
                None if bases is None else bases[rows],
                None if present is None else present[rows],
            )
            parts.append((int(rows.shape[0]), replies))
        return _results(mode, chosen, members, bucket_of, parts, present, tables)

    def _evaluate_rows(
        self,
        fit: Any,
        mode: str,
        points: np.ndarray,
        chosen: tuple,
        request: Request,
        cut: Any,
        retry: tuple,
        longest: float,
        rungs: Any,
        bases: Any,
        present: Any,
    ) -> list:
        """Every process's reply for one group of members, in rank order."""
        points = np.ascontiguousarray(points)
        header = {
            "message": "EVALUATE",
            "cut": cut,
            "fit": fit,
            "mode": mode,
            "chosen": chosen,
            "order": request.order,
            "rungs": rungs,
            "first_steps": bases,
            "units": present,
            "retry": (tuple(retry[0]), tuple(float(f) for f in retry[1])),
            "points_shape": points.shape,
            "points_dtype": points.dtype.str,
        }
        transport = self.transport
        transport.broadcast(header)
        transport.broadcast_array(points)
        own = self.local_evaluate(header, points)
        placement = self.placement
        most = max(
            sum(1 for lane in chosen if placement.rank_of.get(lane) == rank)
            for rank in range(transport.size)
        )
        units = int(points.shape[0]) * max(most, 1)
        replies = [own] + transport.replies(
            "EVALUATE", transports.evaluation_deadline(longest, units)
        )
        for rank, (reply, _) in enumerate(replies):
            error = reply.get("error")
            if error is not None:
                name, text = error
                kind = BackendError if name == "BackendError" else ValueError
                raise kind(f"process {rank}: {text}")
        return replies

    # Prepare --------------------------------------------------------------------------

    def prepare(self, lanes: Any, *, cut: Any, modes: Any, setup: Mapping) -> Any:
        """Compile every process's programs of ``modes`` in two phases; one gathered report.

        Args:
            lanes: The study's lanes without inputs, for the plan of ``cut``.
            cut: ``None``, or ``(members, chunk)`` of a study cut for a
                population.
            modes: The modes to compile; ``None`` takes the default modes.
            setup: The seconds of the stages of lowering the study.

        Returns:
            The [PrepareReport][diff_soup.inference.engine.PrepareReport], one
            record per process and program, and the placement.

        Raises:
            ValueError: If a mode is not a mode of the lane kernel or a
                bucket's solver cannot run it, or a program needs more device
                memory than the budget in any process; the error names every
                such program with its process, the memory it needs and the
                widest chunk that fits.
        """
        self._check_open()
        modes = preparing.default_modes(lanes) if modes is None else modes
        modes = _batched._check_modes(lanes, modes)
        assignment = {
            int(bucket.index): self.placement.compiling_rank(int(bucket.index))
            for bucket in lanes.plan.buckets
        }
        header = {
            "message": "PREPARE",
            "cut": cut,
            "modes": modes,
            "budget_share": None,
            "assignment": assignment,
        }
        start = time.perf_counter()
        transport = self.transport
        transport.broadcast(header)
        first = self.local_prepare(header, "A")
        transport.barrier("PREPARE", transports.prepare_deadline())
        second = self.local_prepare(header, "B")
        replies = [({"records": first[0] + second[0], "refusals": first[1] + second[1]}, ())]
        replies += transport.replies("PREPARE", transports.prepare_deadline())
        seconds = time.perf_counter() - start
        records = [record for reply, _ in replies for record in reply["records"]]
        refusals = [refusal for reply, _ in replies for refusal in reply["refusals"]]
        if refusals:
            raise ValueError(_refusals_text(refusals))
        position = {mode: index for index, mode in enumerate(modes)}
        records.sort(key=lambda record: (record.bucket, position[record.mode], record.rank))
        for bucket in lanes.plan.buckets:
            for mode in modes:
                key = _batched.ProgramKey(bucket.full_identity, mode, int(bucket.chunk[mode].width))
                lanes.programs[(int(bucket.index), mode)] = Remote(key)
        stages = dict(setup)
        for record in records:
            if record.cache == preparing.REUSED:
                continue
            spent = (record.trace_seconds, record.lowering_seconds, record.compile_seconds)
            for stage, value in zip(("trace", "lowering", "compile"), spent):
                stages[f"rank {record.rank} bucket {record.bucket} {record.mode} {stage}"] = value
        return preparing.PrepareReport(
            records=tuple(records),
            stages=types.MappingProxyType(stages),
            seconds=seconds,
            placement=self.placement,
        )

    # The private tier and the end of the run -----------------------------------------

    def private(self, message: str, directory: Any, attempt: int) -> None:
        """Send ``PRIVATE_SAVE`` or ``PRIVATE_LOAD``: nothing is written or read yet."""
        self._check_open()
        if message not in ("PRIVATE_SAVE", "PRIVATE_LOAD"):
            raise ValueError(
                f"the private tier's messages are PRIVATE_SAVE and PRIVATE_LOAD; got {message!r}."
            )
        self.transport.broadcast(
            {"message": message, "directory": str(directory), "attempt": int(attempt)}
        )
        self.transport.replies(message, transports.reply_deadline())

    def shutdown(self) -> None:
        """Send ``SHUTDOWN`` and wait until every worker has acknowledged it. Idempotent.

        When this returns, every worker has freed its device memory and run
        its exit handlers, and goes on to finalize MPI and exit 0. A worker
        that has not acknowledged within the reply deadline aborts the run.
        """
        if self.closed:
            return
        self.closed = True
        self.transport.broadcast({"message": "SHUTDOWN"})
        self.release()
        self.transport.acknowledgements("SHUTDOWN", transports.reply_deadline())


def _refusals_text(refusals: list) -> str:
    fits = [int(refusal["fits"]) for refusal in refusals]
    listed = "; ".join(
        f"process {r['rank']}: the {r['mode']} program of bucket {r['bucket']} (identity "
        f"{r['identity']}) at W = {r['chunk']} needs {r['needs']} bytes of device memory, more "
        f"than the budget of {r['budget']} bytes"
        for r in refusals[:8]
    )
    more = "" if len(refusals) <= 8 else f"; and {len(refusals) - 8} more"
    narrowest = min(fits)
    remedy = (
        f"the largest power-of-two width that fits in every process is {narrowest}: lower the "
        f"study with ds.lower(study, chunk={narrowest}), or take "
        f"LoweredStudy.for_population(members, chunk={narrowest})"
        if narrowest
        else "a program of one unit per call does not fit either, so no chunk= makes it fit"
    )
    return f"{len(refusals)} program(s) do not fit their device ({listed}{more}); {remedy}."


def _results(
    mode: str,
    chosen: tuple,
    members: np.ndarray,
    bucket_of: np.ndarray,
    parts: list,
    present: Any,
    tables: Any,
) -> Results:
    """The results of every group of members, every process's reply placed in canonical order."""
    shape = (int(members.shape[0]), len(chosen))
    column = {lane: index for index, lane in enumerate(chosen)}
    dtypes: dict = {}
    for _, replies in parts:
        for reply, _ in replies:
            if reply.get("lane_ids"):
                for name in ("kind", "t_reached", "rung", "cost", "rows"):
                    dtypes.setdefault(name, np.asarray(reply[name]).dtype)
                for name in _WORK:
                    dtypes.setdefault(name, np.asarray(reply["work"][name]).dtype)
    kind = np.full(shape, ABSENT, dtype=dtypes.get("kind", np.int32))
    t_reached = np.full(shape, np.nan, dtype=dtypes.get("t_reached", np.float64))
    rung = np.zeros(shape, dtype=dtypes.get("rung", np.int32))
    cost = np.zeros(shape, dtype=dtypes.get("cost", np.float64))
    rows = np.zeros(shape, dtype=dtypes.get("rows", np.int64))
    work = {name: np.zeros(shape, dtype=dtypes.get(name, np.int64)) for name in _WORK}
    lane_seconds = np.zeros(shape, dtype=np.float64)
    normal_diagonal: Any = None
    seconds: dict = {}
    batches: dict = {}
    retries = []
    summed: Any = None
    group_sums = []
    start = 0
    for count, replies in parts:
        total = None
        here = slice(start, start + count)
        for reply, arrays in replies:
            ids = reply.get("lane_ids", ())
            if not ids:
                continue
            columns = np.asarray([column[lane] for lane in ids], dtype=np.int64)
            kind[here, columns] = reply["kind"]
            t_reached[here, columns] = reply["t_reached"]
            rung[here, columns] = reply["rung"]
            cost[here, columns] = reply["cost"]
            rows[here, columns] = reply["rows"]
            lane_seconds[here, columns] = reply["lane_seconds"]
            if "normal_diagonal" in reply:
                diagonal = np.asarray(reply["normal_diagonal"], dtype=np.float64)
                if normal_diagonal is None:
                    normal_diagonal = np.zeros(shape + diagonal.shape[-1:], dtype=np.float64)
                normal_diagonal[here, columns] = diagonal
            for name in _WORK:
                work[name][here, columns] = reply["work"][name]
            for bucket, wall in reply["seconds"].items():
                seconds[int(bucket)] = seconds.get(int(bucket), 0.0) + float(wall)
            for bucket, batch in reply["batches"].items():
                units, width, calls = batch
                held = batches.get(int(bucket))
                batches[int(bucket)] = (
                    Batch(units=int(units), width=int(width), calls=int(calls))
                    if held is None
                    else Batch(held.units + int(units), held.width, held.calls + int(calls))
                )
            for record in reply["retries"]:
                entry = dict(record)
                entry["member"] = int(members[start + int(record["member"])])
                retries.append(entry)
            if "levels" in reply:
                level_kind, specs = reply["levels"]
                flat = arrays[0]
                leaves, offset = [], 0
                for leaf_shape, dtype in specs:
                    size = int(np.prod(leaf_shape))
                    leaves.append(flat[offset : offset + size].reshape(leaf_shape).astype(dtype))
                    offset += size
                # The partial tables of every process, added in rank order.
                total = leaves if total is None else [a + b for a, b in zip(total, leaves)]
                summed = (level_kind, tables.depth)
        group_sums.append((total, count))
        start += count
    levels = None
    if mode in _REDUCED and summed is not None:
        level_kind, depth = summed
        template = next(total for total, _ in group_sums if total is not None)
        stacked = []
        for index, leaf in enumerate(template):
            pieces = [
                np.zeros((count,) + leaf.shape[1:], leaf.dtype) if total is None else total[index]
                for total, count in group_sums
            ]
            stacked.append(np.concatenate(pieces, axis=0))
        levels = _unpack_levels(level_kind, depth, stacked)
    if present is not None:
        left_out = ~np.asarray(present, dtype=bool)
        kind[left_out] = ABSENT
    order = {lane: index for index, lane in enumerate(chosen)}
    retries.sort(key=lambda entry: (int(entry["member"]), order[entry["lane"]]))
    return Results(
        mode=mode,
        lane_ids=tuple(chosen),
        members=np.asarray(members, dtype=np.int64),
        bucket=np.asarray(bucket_of, dtype=np.int64),
        status=Status(kind=kind, t_reached=t_reached, rung=rung),
        cost=cost,
        rows=rows,
        normal=None,
        gradient=None,
        observables=None,
        work=Work(**work),
        seconds=types.MappingProxyType(dict(sorted(seconds.items()))),
        retries=tuple(retries),
        batches=types.MappingProxyType(dict(sorted(batches.items()))),
        levels=levels,
        lane_seconds=lane_seconds,
        normal_diagonal=normal_diagonal,
    )


__all__ = [
    "WIRE_BUDGET",
    "Coordinator",
    "Remote",
    "Shard",
    "Worker",
    "plan_fingerprint",
]
