"""Optimization progress callbacks.

Callbacks receive a [StepInfo][diff_soup.inference.fitting.result.StepInfo]
after each optimizer step and can log, checkpoint, or abort the
optimization.
"""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

import numpy as np

from diff_soup.inference.fitting.result import (
    OptResult,
    StepInfo,
    TerminationStatus,
    TrustRegionReport,
)


class StopFit(Exception):
    """Raised from `Callback.on_step` to end a fit after the current step, with status `STOPPED`.

    Every built-in solver catches it except `Optax`.
    """


class Callback(ABC):
    """Base class for a callback that the optimizer calls during a fit.

    A subclass implements `on_step`, and the other methods do nothing by default. Raise
    [StopFit][diff_soup.inference.StopFit] from `on_step` to end the fit early.
    """

    def bind(self, problem: Any) -> None:
        """Receive the problem before the fit starts.

        Args:
            problem: The problem passed to `ds.inference.optsolve`.
        """
        return None

    def on_start(self, n_params: int, n_residuals: int, max_steps: int) -> None:
        """Called once before the first step.

        Args:
            n_params: The number of fitted values.
            n_residuals: The number of residual rows, or `0` when the solver has none.
            max_steps: The optimizer's step budget.
        """
        return None

    @abstractmethod
    def on_step(self, info: StepInfo) -> None:
        """Called after every step.

        Args:
            info: The step's record, with `step`, `cost`, `grad_norm`, `params` and `accepted` among its fields.
        """
        ...

    def on_end(self, result: OptResult) -> None:
        """Called once with the result after the last step.

        Args:
            result: The [OptResult][diff_soup.inference.OptResult] of the fit.
        """
        return None


class LogCallback(Callback):
    """Prints one line per step with the cost, the gradient norm and the step norm.

    Args:
        log_file: A path that also receives the lines, overwritten when it exists.
    """

    def __init__(self, log_file: Optional[str] = None):
        """Open the optional log file alongside stdout.

        Args:
            log_file: Path to a writable text file. The parent directory is
                created if necessary. ``None`` disables file logging.
        """
        self._log_file = None
        if log_file is not None:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            self._log_file = open(log_file, "w")

    def _print(self, msg: str):
        """Emit ``msg`` to stdout and (if configured) the log file."""
        print(msg, flush=True)
        if self._log_file is not None:
            self._log_file.write(msg + "\n")
            self._log_file.flush()

    def on_start(self, n_params: int, n_residuals: int, max_steps: int) -> None:
        """Log the problem dimensions before the first optimizer step."""
        self._print(
            f"  Optimizer: {n_params} params, {n_residuals} residuals, max_steps={max_steps}"
        )

    def on_step(self, info: StepInfo) -> None:
        """Log a one-line summary of this optimizer step."""
        tag = "accept" if info.accepted else "REJECT"
        lm_str = ""
        if info.rho is not None:
            lm_str = f"  rho={info.rho:+.3f}  lam={info.lam:.2e}"
        reason = ""
        if not info.accepted and info.rejection_reason:
            reason = f"  [{info.rejection_reason}]"
        self._print(
            f"    step {info.step:3d}  cost={info.cost:.6e}  |grad|={info.grad_norm:.2e}"
            f"  |d|={info.delta_norm:.2e}{lm_str}  {tag}  ({info.wall_time:.1f}s){reason}"
        )

    def on_end(self, result: Any) -> None:
        """Close the log file when the optimizer finishes."""
        if self._log_file is not None:
            self._log_file.close()


_CHECKPOINT_PREFIX = "step-"
_ATTEMPT_PREFIX = "attempt-"
_KEEP = ("last", "best", "all")
_EVENTS_FILE = "events.jsonl"
_COUNTERS = (
    "residual_evals",
    "jacobian_evals",
    "trial_cost_evals",
    "statistics_evals",
    "trial_predicate_rejections",
    "invalid_pde_trials",
    "damping_only_resolves",
    "block_solves",
)


def _jsonable(value: Any) -> Any:
    """Plain JSON for a step record's value."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return str(value)


def _step_fields(info: StepInfo) -> dict:
    """The entries of a step's event that are read from its record as they are."""
    return {
        "cost": float(info.cost),
        "grad_norm": float(info.grad_norm),
        "delta_norm": float(info.delta_norm),
        "rho": None if info.rho is None else float(info.rho),
        "lam": None if info.lam is None else float(info.lam),
        "radius": None if info.radius is None else float(info.radius),
        "wall_time_s": float(info.wall_time),
        "rejection_reason": info.rejection_reason,
        "failed_owners": (
            None if info.failed_owners is None else [str(o) for o in info.failed_owners]
        ),
    }


def _attempt_files(directory: Any) -> tuple[Path, ...]:
    """The ``attempt-NNNNNN.npz`` files in ``directory``, oldest attempt first."""
    root = Path(directory)
    if not root.is_dir():
        return ()
    return tuple(sorted(root.glob(f"{_ATTEMPT_PREFIX}*.npz")))


def checkpoint_files(directory: Any) -> tuple[Path, ...]:
    """The resumable results in ``directory`` written by ``CheckpointCallback``, oldest first.

    These are the ``attempt-NNNNNN.npz`` files of a solver that resumes its
    state, or, when the directory holds none, the ``step-NNNNNN.npz`` files of
    the other solvers.
    """
    attempts = _attempt_files(directory)
    if attempts:
        return attempts
    root = Path(directory)
    if not root.is_dir():
        return ()
    return tuple(sorted(root.glob(f"{_CHECKPOINT_PREFIX}*.npz")))


def last_checkpoint_event(directory: Any) -> Optional[dict]:
    """The last step record in ``directory``'s event log, or ``None`` without one."""
    path = Path(directory) / _EVENTS_FILE
    if not path.is_file():
        return None
    last = None
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("event") == "step":
                    last = record
    return last


class CheckpointCallback(Callback):
    """Saves a fit to a directory as it runs, so that `ds.inference.optsolve` can resume it.

    Each step appends a line to `events.jsonl`, and every `every`-th step is saved as an
    `.npz` result. Pass the directory as `resume=` or `checkpoint=` to continue the fit.

    Args:
        directory: The directory to write to, created when missing.
        every: Save every `every`-th accepted step, or every `every`-th attempt under a resumable solver such as `TrustRegion`.
        keep: `"last"` keeps the newest saved attempt, `"best"` also the one of lowest cost, and `"all"` every one.
    """

    def __init__(self, directory: Any, every: int = 1, keep: str = "last"):
        """Create ``directory`` and read what a continued run needs from it.

        Args:
            directory: Where the results and ``events.jsonl`` are written.
                Shared across the processes of a run; only the primary
                process writes.
            every: Save a result once for every ``every`` attempts (solvers
                that resume a state) or accepted steps (other solvers). The
                event log records every attempt regardless.
            keep: Which ``attempt-`` files stay: ``"last"``, ``"best"`` or
                ``"all"``.

        Raises:
            ValueError: If ``every`` is below 1 or ``keep`` is none of the
                three.
        """
        from diff_soup._distributed import is_primary

        self.directory = Path(directory)
        self.every = int(every)
        if self.every < 1:
            raise ValueError(f"CheckpointCallback every must be at least 1; got {every}.")
        if not isinstance(keep, str) or keep not in _KEEP:
            raise ValueError(f"CheckpointCallback keep is 'last', 'best' or 'all'; got {keep!r}.")
        self.keep = keep
        self._primary = is_primary()
        if self._primary:
            self.directory.mkdir(parents=True, exist_ok=True)
        last = last_checkpoint_event(self.directory)
        self._step_offset = 0 if last is None else int(last["step"]) + 1
        self._accepted_offset = 0 if last is None else int(last["accepted_count"])
        self._attempt_offset = 0 if last is None else int(last["attempt_count"])
        self._counter_offsets = {
            name: (0 if last is None or last.get(name) is None else int(last[name]))
            for name in _COUNTERS
        }
        self._accepted = 0
        self._attempts = 0
        self._schema = None
        self._x0 = None
        self._max_steps = 0
        self._cost_history: list[float] = []
        self._grad_history: list[float] = []
        # The attempt files keep="best" chooses among, oldest first, with their costs.
        self._candidates: list[tuple[Path, float]] = []
        if keep == "best":
            for path in _attempt_files(self.directory):
                with np.load(path, allow_pickle=False) as data:
                    self._candidates.append((path, float(data["cost"])))

    def bind(self, problem: Any) -> None:
        """Record the problem's parameter schema and start for the saved results."""
        self._schema = getattr(problem, "parameter_schema", None)
        x0 = getattr(problem, "x0", None)
        self._x0 = None if x0 is None else np.asarray(x0, dtype=float)

    def _write_event(self, record: dict) -> None:
        self._write_line({k: _jsonable(v) for k, v in record.items()})

    def _write_line(self, record: dict) -> None:
        with (self.directory / _EVENTS_FILE).open("a") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def on_start(self, n_params: int, n_residuals: int, max_steps: int) -> None:
        """Log the problem dimensions and the offsets this run continues from."""
        self._max_steps = int(max_steps)
        if not self._primary:
            return
        self._write_event(
            {
                "event": "start",
                "n_params": int(n_params),
                "n_residuals": int(n_residuals),
                "max_steps": int(max_steps),
                "step_offset": self._step_offset,
                "accepted_offset": self._accepted_offset,
                "attempt_offset": self._attempt_offset,
            }
        )

    def _record(self, info: StepInfo) -> dict:
        step = int(info.step) + self._step_offset
        accepted_count = (
            self._accepted_offset + self._accepted
            if info.accepted_count is None
            else self._accepted_offset + int(info.accepted_count)
        )
        attempt_count = (
            self._attempt_offset + self._attempts
            if info.attempt_count is None
            else self._attempt_offset + int(info.attempt_count)
        )
        record = {
            "event": "step",
            "step": step,
            "accepted": bool(info.accepted),
            "accepted_count": accepted_count,
            "attempt_count": attempt_count,
            **_step_fields(info),
        }
        for name in _COUNTERS:
            value = getattr(info, name, None)
            record[name] = None if value is None else self._counter_offsets[name] + int(value)
        return record

    def _attempt_record(self, info: StepInfo, state: Any) -> dict:
        """The event of an attempt of a solver that resumes its state, numbered by the solver."""
        record = {
            "event": "step",
            "step": int(info.step),
            "accepted": bool(info.accepted),
            "accepted_count": (
                int(np.asarray(state["accepted"]))
                if info.accepted_count is None
                else int(info.accepted_count)
            ),
            "attempt_count": (
                int(info.step) + 1 if info.attempt_count is None else int(info.attempt_count)
            ),
            **_step_fields(info),
        }
        for name in _COUNTERS:
            value = getattr(info, name, None)
            record[name] = None if value is None else int(value)
        record = {key: _jsonable(value) for key, value in record.items()}
        record["failed_owners"] = (
            None if info.failed_owners is None else [str(owner) for owner in info.failed_owners]
        )
        if "record.failed_lanes" in state:
            lanes = np.ravel(np.asarray(state["record.failed_lanes"]))
            kinds = np.ravel(np.asarray(state["record.failed_kinds"]))
            record["failed_lanes"] = [[str(lane), str(kind)] for lane, kind in zip(lanes, kinds)]
        return record

    def _write_result(self, result: OptResult, final: Path) -> None:
        """Save ``result`` as ``final`` through a temporary file in the same directory."""
        handle, temporary = tempfile.mkstemp(dir=self.directory, suffix=".npz")
        os.close(handle)
        try:
            result.save(temporary)
            os.replace(temporary, final)
        except BaseException:
            if os.path.exists(temporary):
                os.remove(temporary)
            raise

    def _save_attempt(self, info: StepInfo, state: Any) -> None:
        """Write ``attempt-NNNNNN.npz`` for the attempt just made, then apply ``keep``."""
        attempt = int(info.step)
        accepted = int(np.asarray(state["accepted"]))
        cost = float(np.asarray(state["cost"]))
        params = np.asarray(state["point"], dtype=float)
        info_fields = {
            "checkpoint_step": attempt,
            "accepted_count": accepted,
            "attempt_count": attempt + 1,
        }
        if info.lam is not None:
            info_fields["lam"] = float(info.lam)
        work = {
            key[len("work.") :]: value for key, value in state.items() if key.startswith("work.")
        }
        result = OptResult(
            params=params,
            initial_params=params if self._x0 is None else self._x0,
            cost=cost,
            grad_norm=float(info.grad_norm),
            status=TerminationStatus.MAX_STEPS,
            message="checkpoint of a running fit",
            n_steps=attempt + 1,
            max_steps=self._max_steps,
            n_accepted=accepted,
            cost_history=np.asarray(state["cost_history"], dtype=float),
            grad_history=np.asarray(state["grad_history"], dtype=float),
            info=info_fields,
            parameter_schema=self._schema,
            work=MappingProxyType(work) if work else None,
            state=state,
            trust_region=TrustRegionReport(radius_history=np.ravel(state["radius_history"])),
        )
        final = self.directory / f"{_ATTEMPT_PREFIX}{attempt:06d}.npz"
        self._write_result(result, final)
        self._prune(final, cost)

    def _prune(self, newest: Path, cost: float) -> None:
        """Remove the attempt files ``keep`` does not hold, now that ``newest`` is complete."""
        if self.keep == "all":
            return
        kept = {newest.name}
        if self.keep == "best":
            self._candidates = [
                entry for entry in self._candidates if entry[0].name != newest.name
            ] + [(newest, cost)]
            best = None
            for path, value in self._candidates:
                # The newer file wins a tie; a NaN cost is never the best of finite ones.
                if best is None or np.isnan(best[1]) or value <= best[1]:
                    best = (path, value)
            kept.add(best[0].name)
            self._candidates = [entry for entry in self._candidates if entry[0].name in kept]
        for path in _attempt_files(self.directory):
            if path.name not in kept:
                path.unlink(missing_ok=True)

    def _save(self, info: StepInfo, record: dict) -> None:
        params = np.asarray(info.params, dtype=float)
        info_fields = {
            "checkpoint_step": int(record["step"]),
            "accepted_count": int(record["accepted_count"]),
            "attempt_count": int(record["attempt_count"]),
        }
        if info.lam is not None:
            info_fields["lam"] = float(info.lam)
        result = OptResult(
            params=params,
            initial_params=params if self._x0 is None else self._x0,
            cost=float(info.cost),
            grad_norm=float(info.grad_norm),
            status=TerminationStatus.MAX_STEPS,
            message="checkpoint of a running fit",
            n_steps=int(record["step"]) + 1,
            max_steps=self._max_steps,
            n_accepted=int(record["accepted_count"]),
            cost_history=np.asarray(self._cost_history, dtype=float),
            grad_history=np.asarray(self._grad_history, dtype=float),
            info=info_fields,
            parameter_schema=self._schema,
            trust_region=(
                None if info.radius is None else TrustRegionReport(radius_history=(info.radius,))
            ),
        )
        final = self.directory / f"{_CHECKPOINT_PREFIX}{int(record['step']):06d}.npz"
        self._write_result(result, final)

    def on_step(self, info: StepInfo) -> None:
        """Append the attempt's record and save a result when one is due.

        With a state, an ``attempt-`` file after every ``every``-th attempt;
        without one, a ``step-`` file after every ``every``-th accepted step.
        """
        state = info.state
        if state is not None:
            if not self._primary:
                return
            self._write_line(self._attempt_record(info, state))
            if (int(info.step) + 1) % self.every == 0:
                self._save_attempt(info, state)
            return
        self._attempts += 1
        if info.accepted:
            self._accepted += 1
            self._cost_history.append(float(info.cost))
            self._grad_history.append(float(info.grad_norm))
        if not self._primary:
            return
        record = self._record(info)
        self._write_event(record)
        if info.accepted and self._accepted % self.every == 0:
            self._save(info, record)

    def on_end(self, result: OptResult) -> None:
        """Log how the optimizer stopped."""
        if not self._primary:
            return
        self._write_event(
            {
                "event": "end",
                "status": str(result.status.value),
                "cost": float(result.cost),
                "n_steps": int(result.n_steps),
                "n_accepted": int(result.n_accepted),
                "wall_time_s": float(result.wall_time),
            }
        )
