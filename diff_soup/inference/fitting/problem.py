"""Fit declared experiments with shared or owner-varying Recipe parameters."""

from __future__ import annotations

import atexit
import collections.abc
import copy
import dataclasses
import math
import os
import time
import types
import weakref
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup._warnings import warn_user
from diff_soup.inference.engine import _batched, _transport
from diff_soup.inference.engine._ledger import Ledger
from diff_soup.inference.engine._levels import (
    LevelTables,
    joint_gradient,
    joint_matrix,
    member_row,
    reduce_member_gradients,
    weigh_levels,
)
from diff_soup.inference.engine._plan import (
    Plan,
    _chunk_width,
    _member_count,
    build_plan,
    population_plan,
)
from diff_soup.inference.engine._prepare import SETUP_STAGES, PrepareReport
from diff_soup.inference.engine._prepare import prepare as prepare_lanes
from diff_soup.inference.engine._reducers import TreeTables
from diff_soup.inference.engine._request import Request
from diff_soup.inference.engine._result import Results
from diff_soup.inference.engine._status import KINDS
from diff_soup.inference.fitting.result import ExperimentFit
from diff_soup.inference.statistics.estimation import (
    MAP,
    MLE,
    EstimationProblem,
    lower_estimation,
)
from diff_soup.inference.statistics.hierarchy import Hierarchy
from diff_soup.inference.statistics.measurement import Experiment
from diff_soup.inference.statistics.study import Study
from diff_soup.problem.parameters import ParameterSet

_RUNS: list = []
"""One entry once this process lowered a study in a run of several processes."""


class InferenceProblem(EstimationProblem):
    """An experiment or a study paired with the objective a fit minimizes.

    Pass it to `ds.inference.optsolve` or to the statistics methods. `ds.lower(study)` builds
    the batched form of the same problem.

    Args:
        experiments: An `Experiment`, a sequence of them, or a `Study`, which already declares its parameters and clusters.
        parameters: The handles the fit moves: one handle, a tuple of them, a `ParameterSet` or a `Hierarchy`.
        objective: [MLE][diff_soup.inference.MLE], or [MAP][diff_soup.inference.MAP], which adds the declared priors.
        clusters: The cluster label of each experiment, for a parameter with one value per cluster.
        layout: The placement of the experiments on a device mesh.
        trajectory: A `SingleShoot` or `Multishoot` applied to every experiment.
        regularization: A penalty on the parameters added to the objective, such as `Tikhonov`.
        trial_predicate: A function of the parameters returning whether a trial point may be evaluated.
    """

    def __init__(
        self,
        experiments: Any,
        *,
        parameters: Any = None,
        objective: MLE | MAP = MLE(),
        clusters: Any = None,
        layout: Any = None,
        trajectory: Any = None,
        regularization: Any = None,
        trial_predicate: Any = None,
    ):
        """Compile experiments, handle ownership, and the chosen objective.

        Args:
            experiments: An Experiment, a sequence of Experiments, or a Study.
            parameters: Recipe handles fitted by every experiment, a
                ``ParameterSet(shared=, varying=)``, or a ``Hierarchy``.
            objective: ``MLE()`` or ``MAP(...)``.
            clusters: Cluster labels for ``varying`` parameters of a
                ``ParameterSet``; left out with a ``Hierarchy``.
            layout: Optional device placement of the experiment batch.
            trajectory: Optional trajectory strategy replacing every
                experiment's own (``SingleShoot()`` or ``Multishoot(...)``).
            regularization: Optional regularizer on the fitted vector;
                ``Tikhonov(lambda_=..., on=handle)`` applies to one handle's
                columns, per owner when it varies.
            trial_predicate: Optional JAX-traceable ``x -> bool`` over the
                fitted vector; a trial the predicate rejects is never solved.
        """
        if isinstance(experiments, Study):
            if parameters is not None or clusters is not None:
                raise TypeError("A Study already declares parameters and clusters.")
            study = experiments
        else:
            if isinstance(experiments, Experiment):
                experiments = (experiments,)
            else:
                try:
                    experiments = tuple(experiments)
                except TypeError as exc:
                    raise TypeError(
                        "InferenceProblem requires an Experiment, a sequence of Experiments, or a Study."
                    ) from exc
            if not isinstance(parameters, (ParameterSet, Hierarchy)):
                if hasattr(parameters, "identity"):
                    parameters = (parameters,)
                if parameters is None:
                    raise TypeError("InferenceProblem requires parameters= with Recipe handles.")
                parameters = ParameterSet(shared=parameters)
            study = Study(experiments, parameters=parameters, clusters=clusters, independent=True)
        if trajectory is not None:
            # A hierarchy study derives its clusters from the experiments' members.
            study = Study(
                tuple(dataclasses.replace(e, trajectory=trajectory) for e in study.experiments),
                parameters=study.parameters,
                clusters=None if isinstance(study.parameters, Hierarchy) else study.clusters,
                independent=study.independent,
            )
        super().__init__(
            study,
            objective,
            layout=layout,
            regularization=regularization,
            trial_predicate=trial_predicate,
        )
        object.__setattr__(self, "layout", layout)

    @property
    def placement(self) -> dict:
        """A mapping from experiment id to the pair (index of the batched solve, process index) that evaluates it."""
        import jax

        lowered = self._inference_problem
        lanes = getattr(lowered, "placement", None)
        if lanes is None:
            lanes = ((0, jax.process_index()),) * len(self.study.experiment_ids)
        return {
            identifier: (int(lane), int(process))
            for identifier, (lane, process) in zip(self.study.experiment_ids, lanes)
        }

    def inventory(self) -> dict:
        """Describe the layout of the parameters and the data as one JSON-serializable dict.

        Returns:
            A dict with `n_parameters`, the `residual_rows` by kind, the `parameter_columns` by group, the clusters as `owners`, and one entry per experiment.
        """
        import jax

        joint = self._inference_problem
        ids = tuple(self.study.experiment_ids)
        kinds = {}
        states = {}
        children = getattr(joint, "problems", (joint,))
        for identifier, child in zip(ids, children):
            kinds[identifier] = dict(child.residual_row_kinds)
        total = {
            key: sum(kind[key] for kind in kinds.values())
            for key in ("observation", "continuity", "state_prior")
        }
        regularization = 0
        numeric = self.joint_view() if hasattr(joint, "owner_ids") else None
        source = numeric if numeric is not None else self
        try:
            regularization = int(
                jax.eval_shape(source.regularization.residual, jnp.asarray(self.x0)).shape[0]
            )
        except Exception:  # noqa: BLE001 -- a regularizer without a traceable residual reports no rows
            regularization = 0
        total["regularization"] = regularization
        total["total"] = sum(total.values())
        columns = {}
        owners = {}
        placement = self.placement
        record = {
            "n_parameters": int(self.n_params),
            "experiments": {},
            "residual_rows": total,
        }
        if hasattr(joint, "owner_ids"):
            layout = joint.ownership_layout
            columns = {
                name: len(cols)
                for name, cols in zip(joint.param_group_names, joint.param_group_cols)
            }
            for owner, owner_columns in zip(joint.owner_ids, layout.owner_columns):
                owners[str(owner)] = {"width": len(owner_columns), "experiments": []}
            for identifier, child_index in zip(ids, range(len(children))):
                lo, hi = layout.state_slices[child_index]
                states[identifier] = int(hi - lo)
                owner = joint.owner_ids[joint.child_owner_indices[child_index]]
                owners[str(owner)]["experiments"].append(identifier)
        for identifier in ids:
            lane, process = placement[identifier]
            record["experiments"][identifier] = {
                **kinds[identifier],
                "state_coordinates": int(states.get(identifier, 0)),
                "lane": int(lane),
                "process": int(process),
            }
        record["parameter_columns"] = columns
        record["owners"] = owners
        return record

    def with_start(self, start: Any) -> InferenceProblem:
        """This problem, compiled once, started at another point.

        Args:
            start: A flat vector in this problem's column order, a mapping from every fitted
                handle to its value, or an `OptResult` whose columns hold the same handles.
                A fit under another solver tolerance or starting value is accepted.

        Returns:
            The problem with `x0` set to `start`.

        Raises:
            ValueError: If `start` has another parameter layout or shape.
        """
        from diff_soup.inference.fitting._layout import resolve_point

        values = np.asarray(resolve_point(self, start, where="InferenceProblem.with_start"))
        if values.shape != self.x0.shape:
            raise ValueError(
                f"InferenceProblem.with_start: start has shape {values.shape}; expected "
                f"{self.x0.shape}."
            )
        result = copy.copy(self)
        object.__setattr__(result, "_x0", values)
        return result

    def finalize_opt_result(self, result, *, optimizer):
        """Attach the objective and each experiment's fitted parameter values."""
        result = super().finalize_opt_result(result, optimizer=optimizer)
        children = getattr(self._inference_problem, "child_results", None)
        reconstructions = children(result.params) if callable(children) else (result,)
        experiments = {}
        for i, (identifier, child) in enumerate(zip(self.study.experiment_ids, reconstructions)):
            owners = {
                handle.name: self.study.clusters[cluster][i]
                for cluster, handles in self.study.parameters.varying.items()
                for handle in handles
            }
            experiments[identifier] = ExperimentFit(
                opt_params=child.opt_params,
                parameter_schema=self.parameter_schema,
                params=result.params,
                owners=owners,
                window_states=getattr(child, "window_states", None),
            )
        return dataclasses.replace(result, experiments=experiments)


def _evaluation_noise(study: Study, given: Any) -> float | None:
    """The level ``given``, checked, else the largest ``rtol`` of the experiments' solvers."""
    if given is not None:
        try:
            value = float(given)
        except (TypeError, ValueError):
            value = math.nan
        if isinstance(given, bool) or not (math.isfinite(value) and value > 0.0):
            raise ValueError(
                f"evaluation_noise must be a positive finite number or None; got {given!r}."
            )
        return value
    tolerances = []
    for experiment in study.experiments:
        rtol = getattr(getattr(experiment, "solver", None), "rtol", None)
        if rtol is not None:
            tolerances.append(float(np.max(np.asarray(rtol, dtype=np.float64))))
    return max(tolerances) if tolerances else None


RETRY_KINDS: tuple[str, ...] = ("nonlinear_diverged", "step_collapse")
"""The failure kinds a lowered study solves again by default."""

RETRY_FIRST_STEP_FACTORS: tuple[float, ...] = (0.1, 1e-3)
"""The default factors of the first-step rungs, in the order they are tried."""


def _setting(value: Any, name: str, what: str) -> tuple:
    """``value`` as a tuple, refusing a single string or a value that is not a sequence."""
    if isinstance(value, (str, bytes)):
        raise ConfigurationError(
            f"{name}= is a tuple of {what}; got the single string {value!r}. Write "
            f"{name}=({value!r},)."
        )
    try:
        return tuple(value)
    except TypeError:
        raise ConfigurationError(f"{name}= is a tuple of {what}; got {value!r}.") from None


def _retry_settings(kinds: Any, factors: Any, tolerance: Any) -> tuple:
    """The three retry settings of a lowered study, checked."""
    failures = tuple(kind for kind in KINDS if kind != "ok")
    kinds = _setting(kinds, "retry_kinds", "failure kinds")
    for kind in kinds:
        if kind == "ok":
            raise ConfigurationError(
                "retry_kinds= names 'ok', the kind of a lane that solved, and only a failure is "
                f"solved again; name failure kinds, among {failures}."
            )
        if kind not in failures:
            raise ConfigurationError(
                f"retry_kinds= names the kind {kind!r}, which no lane reports; name failure "
                f"kinds, among {failures}."
            )
        if kinds.count(kind) > 1:
            raise ConfigurationError(
                f"retry_kinds= names the kind {kind!r} more than once; name each kind once."
            )
    factors = _setting(factors, "retry_first_step_factors", "positive numbers")
    checked = []
    for factor in factors:
        try:
            if isinstance(factor, (bool, np.bool_)):
                raise TypeError
            value = float(factor)
        except (TypeError, ValueError):
            value = math.nan
        if not (math.isfinite(value) and value > 0.0):
            raise ConfigurationError(
                f"retry_first_step_factors= holds the factor {factor!r}; a factor scales the "
                "first step of a lane's solve, so each must be a positive finite number."
            )
        checked.append(value)
    tolerance = _setting(tolerance, "retry_tolerance_factors", "numbers")
    if tolerance:
        raise ConfigurationError(
            f"retry_tolerance_factors={tolerance!r} asks to solve a failed lane again with "
            "tighter tolerances, which the lanes of a lowered study do not offer: a tolerance "
            "is part of a compiled program. Leave retry_tolerance_factors=() and solve failed "
            "lanes again from a smaller first step with retry_first_step_factors=."
        )
    return kinds, tuple(checked), tolerance


def _on_the_host(function: Any, point: Any, shapes: Any) -> Any:
    """``function`` of a point's numbers: called at once, or through ``jax.pure_callback`` when traced.

    ``function`` takes points ``(..., n)`` and returns numbers per point, so
    under ``jax.vmap`` the callback receives every point of the batch at once.
    """
    if isinstance(point, jax.core.Tracer):
        return jax.pure_callback(function, shapes, point, vmap_method="expand_dims")
    return jax.tree.map(jnp.asarray, function(np.asarray(point)))


class ValueAndGradient(NamedTuple):
    """The objective of a lowered study and its gradient at many points, from one request.

    [LoweredStudy.value_and_gradient][diff_soup.inference.LoweredStudy.value_and_gradient]
    returns it.

    Attributes:
        values: ``(P,)``, or ``(Q, P)`` with weights: the objective at every
            point; NaN at a point where a lane of nonzero weight failed.
        gradients: ``(P, n)``, or ``(Q, P, n)`` with weights: the gradient of
            the objective in the coordinates of ``x0``; every entry NaN where
            the value is.
        failures: One record per failed lane of nonzero weight, in canonical
            order: ``member`` (the row of the weights, 0 without weights),
            ``point``, ``lane``, ``kind`` (a name of
            [KINDS][diff_soup.inference.engine.KINDS]), ``t_reached`` and
            ``rung`` (the last rung tried).
        results: The [Results][diff_soup.inference.engine.Results] of the one
            ``gradient`` request: work, walls, batches and retries.
    """

    values: np.ndarray
    gradients: np.ndarray
    failures: tuple
    results: Results


def _measured_form(measured: Any) -> None:
    """Refuse a ``measured=`` that is none of the forms ``ds.lower`` takes; nothing is read."""
    if measured is None or isinstance(measured, (Ledger, str, os.PathLike)):
        return
    if isinstance(measured, collections.abc.Mapping):
        return
    raise TypeError(
        "measured= is None, a checkpoint directory or file (a path), a Ledger, or a mapping of "
        f"lane id to the seconds one evaluation of the lane took; got {type(measured).__name__}."
    )


def _measured_seconds(measured: Any, lanes: tuple) -> dict | None:
    """``measured=`` as lane id to seconds per evaluation, for the planned ``lanes`` it names.

    A path is read with ``Ledger.load``; a ledger gives its mean seconds per
    evaluation. Lanes are matched by their ids as text, which is how a
    checkpoint saves them. ``None`` when nothing is measured.
    """
    if measured is None:
        return None
    if isinstance(measured, (str, os.PathLike)):
        measured = Ledger.load(measured)
    if isinstance(measured, Ledger):
        measured = measured.seconds_per_evaluation()
    by_text = {str(lane): float(seconds) for lane, seconds in dict(measured).items()}
    found = {lane: by_text[str(lane)] for lane in lanes if str(lane) in by_text}
    return found or None


class LoweredStudy(InferenceProblem):
    """A study prepared for fitting by `ds.lower(study)`, with its experiments' solves compiled and batched.

    `ds.inference.optsolve` and the statistics methods take it in place of an
    `InferenceProblem`, and `ds.inference.TrustRegion` fits only this form. A failed solve is
    retried from a smaller first step.
    """

    def __init__(
        self,
        study: Study,
        *,
        objective: MLE | MAP = MLE(),
        layout: Any = None,
        regularization: Any = None,
        trial_predicate: Any = None,
        trajectory: Any = None,
        chunk: int | None = None,
        device: Any = None,
        evaluation_noise: float | None = None,
        retry_kinds: Any = RETRY_KINDS,
        retry_first_step_factors: Any = RETRY_FIRST_STEP_FACTORS,
        retry_tolerance_factors: Any = (),
        measured: Any = None,
    ):
        """Lower a study: its experiments, the plan of its lanes, and their numbers.

        Args:
            study: The [Study][diff_soup.inference.Study].
            objective: ``MLE()`` or ``MAP(...)``.
            layout: Optional device placement, also of the experiment batch
                for the routes of ``InferenceProblem``. A
                [Layout][diff_soup.Layout] whose mesh has a ``data`` axis of
                ``n > 1`` devices places the lanes over those devices (every
                call of every program is split over them); without one the
                lanes run on ``device``. ``Layout.processes`` is the number of
                processes a run of several processes is expected to have.
            regularization: Optional regularizer on the fitted vector, as for
                ``InferenceProblem``.
            trial_predicate: Optional JAX-traceable ``x -> bool`` over the
                fitted vector, as for ``InferenceProblem``.
            trajectory: Optional trajectory strategy replacing every
                experiment's own.
            chunk: The units (pairs of population member and lane) one call of
                every compiled program evaluates, taken exactly as given; a
                chunk wider than a bucket's units fills its call with copies
                whose results are dropped. ``None`` takes, per bucket, the
                smallest power of two that holds its units, at most 1,024.
            device: The JAX device the lanes run on: their compiled programs,
                their inputs, and the tables and level systems their results
                are reduced into. ``None`` takes this process's first device,
                ``jax.local_devices()[0]``. With a layout that places the
                lanes over a mesh, ``device`` must be a device of that mesh or
                ``None`` (its first device).
            evaluation_noise: The largest relative noise with which the lanes
                evaluate the cost (the noise of the cost's evaluation, not the
                data's), a positive number; ``None`` takes the largest ``rtol``
                of the experiments' solvers. See
                [evaluation_noise][diff_soup.inference.LoweredStudy.evaluation_noise].
            retry_kinds: The failure kinds (names of
                [KINDS][diff_soup.inference.engine.KINDS] other than ``"ok"``)
                after which a lane is solved again from a smaller first step;
                ``()`` solves no lane again.
            retry_first_step_factors: Positive factors, one per rung in the
                order they are tried: rung ``r`` starts the failed lane's solve
                from ``retry_first_step_factors[r - 1]`` times the first step
                its own solve took; ``()`` solves no lane again.
            retry_tolerance_factors: Must be ``()``: solving a lane again with
                tighter tolerances would need another compiled program, so the
                lanes retry from a smaller first step only.
            measured: In a run of several processes, balance the lanes over
                the processes by the seconds a previous run measured, from its
                checkpoint: the checkpoint directory of a fit or a bootstrap
                (or one of its ``attempt-`` or ``population-`` files), a
                [Ledger][diff_soup.inference.engine.Ledger], or a mapping of
                lane id to the seconds one evaluation of the lane took; a lane
                without a measurement is balanced by its modelled cost.
                ``None`` balances every lane by its modelled cost. Ignored in
                a run of one process, so the same script runs everywhere.

        Raises:
            TypeError: If ``study`` is not a ``Study``, ``device`` is not a
                JAX device or ``None``, or ``measured`` is none of the forms
                above.
            ValueError: If ``chunk`` is not a positive integer or ``None``, or
                ``evaluation_noise`` is not a positive finite number or
                ``None``.
            ConfigurationError: If ``retry_kinds`` names a kind that is not a
                failure kind or names one twice, a factor of
                ``retry_first_step_factors`` is not a positive finite number,
                or ``retry_tolerance_factors`` is not empty.
        """
        if not isinstance(study, Study):
            raise TypeError(
                f"LoweredStudy lowers a Study; got {type(study).__name__}. Declare the "
                "experiments in a Study and lower that."
            )
        _chunk_width(chunk)
        if device is not None and not isinstance(device, jax.Device):
            raise TypeError(
                f"device= is a JAX device, such as jax.devices('cpu')[0], or None for the first "
                f"device; got {device!r}."
            )
        noise = _evaluation_noise(study, evaluation_noise)
        retry = _retry_settings(retry_kinds, retry_first_step_factors, retry_tolerance_factors)
        _measured_form(measured)
        start = time.perf_counter()
        super().__init__(
            study,
            objective=objective,
            layout=layout,
            trajectory=trajectory,
            regularization=regularization,
            trial_predicate=trial_predicate,
        )
        lowered_at = time.perf_counter()
        joint = self._inference_problem
        world = _transport.world()
        object.__setattr__(self, "_evaluation_noise", noise)
        object.__setattr__(self, "_retry", retry)
        if world.size > 1:
            self._lower_in_processes(world, joint, chunk, device, start, lowered_at, measured)
            return
        plan = build_plan(self.study, joint, chunk=chunk, device=device)
        planned_at = time.perf_counter()
        lanes = _batched.batched_lanes(plan, joint)
        stacked_at = time.perf_counter()
        seconds = (lowered_at - start, planned_at - lowered_at, stacked_at - planned_at)
        object.__setattr__(self, "_lanes", lanes)
        object.__setattr__(self, "_setup", types.MappingProxyType(dict(zip(SETUP_STAGES, seconds))))
        object.__setattr__(self, "_ledger", Ledger(lanes.lane_ids[: len(lanes.where)]))

    def _lower_in_processes(
        self,
        world: Any,
        joint: Any,
        chunk: Any,
        device: Any,
        start: float,
        lowered_at: float,
        measured: Any = None,
    ) -> None:
        """Plan, place and build this process's lanes in a run of several processes.

        Every process builds the plan and checks its fingerprint against every
        other process's; process 0 places the lanes and every process keeps its
        own. Process 0 keeps the coordinator; every other process serves the
        coordinator's messages from here and exits when the run shuts down.
        The script is written as ``with ds.distributed.run(study) as
        lowered:``; the other processes end with ``SystemExit(0)`` when the
        study is closed.
        """
        from diff_soup.inference.engine import _executor, _placement

        if _RUNS:
            raise RuntimeError(
                "a run of several processes lowers one study: the processes above 0 serve the "
                "first lowered study until it is closed and the run ends; declare every "
                "experiment in one Study and lower it once, or run one process."
            )
        _RUNS.append(True)
        if device is not None:
            warn_user(
                f"device={device!r} is ignored in a run of {world.size} processes: each process "
                "evaluates its lanes on its one visible device, jax.devices()[0].",
                UserWarning,
            )
        transport = _transport.Transport(world)
        plan = build_plan(self.study, joint, chunk=chunk, device=None)
        planned_at = time.perf_counter()
        metadata = _batched.lane_metadata(plan, joint)
        object.__setattr__(self, "_lanes", metadata)
        object.__setattr__(self, "_ledger", Ledger(metadata.lane_ids[: len(metadata.where)]))
        # PLAN_CHECK: every process holds the same plan, or none goes on.
        fingerprints = transport.allgather(_executor.plan_fingerprint(plan))
        differing = [rank for rank, value in enumerate(fingerprints) if value != fingerprints[0]]
        if differing:
            raise RuntimeError(
                f"the processes {differing} built another plan of the study than process 0 "
                f"(plan fingerprints {fingerprints[0][:12]} and "
                f"{fingerprints[differing[0]][:12]}): every process has to lower the same study "
                "with the same settings, library and data."
            )
        # PLACE: process 0 places the lanes; every process keeps its own.
        offered = None
        if transport.rank == 0:
            offered = self._placement_inputs(metadata, measured)
        offered = transport.comm.bcast(offered, root=0)
        placement = _placement.place(
            plan,
            world.size,
            offered["costs"],
            offered["measured"],
            experiments=offered["experiments"],
            members=offered["members"],
        )
        cluster = (_executor.Coordinator if transport.rank == 0 else _executor.Worker)(
            transport, self, joint, metadata, placement
        )
        stacked_at = time.perf_counter()
        seconds = (lowered_at - start, planned_at - lowered_at, stacked_at - planned_at)
        object.__setattr__(self, "_setup", types.MappingProxyType(dict(zip(SETUP_STAGES, seconds))))
        if transport.rank != 0:
            cluster.serve()
        object.__setattr__(self, "_executor", cluster)
        atexit.register(cluster.shutdown)

    def _placement_inputs(self, metadata: Any, measured: Any = None) -> dict:
        """What process 0 places the lanes by: modelled and measured costs, experiments, members.

        ``measured`` is ``ds.lower``'s: a checkpoint path is read here, on
        process 0, and every form becomes lane id to seconds per evaluation for
        the planned lanes it names.
        """
        from diff_soup.inference.engine._layout import lane_layout
        from diff_soup.inference.engine._placement import modelled_costs

        layout = lane_layout(self._inference_problem)
        experiments, members = {}, {}
        for entry in metadata.buckets:
            for lane, child in zip(entry.bucket.lanes, entry.bucket.children):
                experiments[lane] = int(layout.experiments[child])
        try:
            route = self._route()
        except ConfigurationError:
            route = None
        if route is not None:
            for lane, position in route.position.items():
                members[lane] = int(route.nodes[position][0])
        return {
            "costs": modelled_costs(metadata),
            "measured": _measured_seconds(measured, tuple(experiments)),
            "experiments": experiments,
            "members": members,
        }

    @property
    def processes(self) -> int:
        """The number of processes the study's experiments are split over."""
        executor = self.__dict__.get("_executor")
        return 1 if executor is None else int(executor.transport.size)

    def close(self) -> None:
        """Stop the other processes of a run of several processes, or do nothing in a run of one."""
        executor = self.__dict__.get("_executor")
        if executor is not None:
            executor.shutdown()

    def _plan_fingerprint(self) -> str | None:
        """The fingerprint of the plan: lanes, bucket identities, widths; ``None`` without a plan."""
        from diff_soup.inference.engine._executor import plan_fingerprint

        lanes = self.__dict__.get("_lanes")
        return None if lanes is None else plan_fingerprint(lanes.plan)

    @property
    def retry_kinds(self) -> tuple[str, ...]:
        """The failure kinds that are solved again from a smaller first step."""
        return self._retry[0]

    @property
    def retry_first_step_factors(self) -> tuple[float, ...]:
        """The factors of the first step a failed solve is retried with, in the order tried."""
        return self._retry[1]

    @property
    def retry_tolerance_factors(self) -> tuple:
        """Rungs that would tighten the tolerances of a failed lane's solver: always ``()``.

        A tolerance is part of a lane's compiled program, so a tolerance rung
        would compile another program per bucket and mode; the lanes of a
        lowered study retry from a smaller first step only
        ([retry_first_step_factors][diff_soup.inference.LoweredStudy.retry_first_step_factors]).
        """
        return self._retry[2]

    @property
    def evaluation_noise(self) -> float | None:
        """The relative noise of the objective's evaluation, by default the largest `rtol` of the experiments' solvers, or `None`."""
        return self._evaluation_noise

    def plan(self) -> Plan:
        """The plan that groups the experiments' solves into batches."""
        return self._lanes.plan

    def for_population(self, members: int, *, chunk: Any = "auto") -> LoweredStudy:
        """A copy of the study re-planned to evaluate `members` parameter points per request.

        Nothing is lowered again, and the copy reuses this study's compiled solves.

        Args:
            members: The number of points each request evaluates.
            chunk: The number of solves one compiled call evaluates, `"auto"` to choose it, or `None` to return this study.
        """
        members = _member_count(members)
        if chunk is None:
            return self
        if isinstance(chunk, str) and chunk == "auto":
            forced = None
        else:
            try:
                forced = _chunk_width(chunk)
            except ValueError:
                raise ValueError(
                    "chunk= is 'auto', the number of units one call of a program evaluates (a "
                    f"positive integer), or None to keep this study's programs; got {chunk!r}."
                ) from None
        start = time.perf_counter()
        plan = population_plan(self._lanes.plan, members, chunk=forced)
        planned_at = time.perf_counter()
        lanes = _batched.replanned_lanes(self._lanes, plan)
        shared_at = time.perf_counter()
        # Nothing is lowered again: the experiments' stage takes no time.
        seconds = (0.0, planned_at - start, shared_at - planned_at)
        kept = self.__dict__.get("_population_programs")
        if kept is None:
            kept = {}
            object.__setattr__(self, "_population_programs", kept)
        population = copy.copy(self)
        object.__setattr__(population, "_lanes", lanes)
        # In a run of several processes, the processes cut their own lanes the same way.
        object.__setattr__(population, "_cut", (int(members), forced))
        object.__setattr__(
            population, "_setup", types.MappingProxyType(dict(zip(SETUP_STAGES, seconds)))
        )
        # The cut study hands the programs its prepare() holds to this study, which keeps them.
        object.__setattr__(population, "_keeper", kept)
        return population

    def prepare(self, *, modes: Any = None, **options: Any) -> PrepareReport:
        """Compile the study's solves before the first evaluation.

        Args:
            modes: The evaluation modes to compile, such as `"value"`, `"gradient"` or `"blocks"`, or `None` for the default set.
            **options: Not accepted.

        Returns:
            A report of what was compiled and how long each setup stage took.

        Raises:
            TypeError: If any keyword other than `modes` is passed.
        """
        if options:
            raise TypeError(
                f"LoweredStudy.prepare takes modes= only; got {sorted(options)}. The plan "
                "chooses the derivative width and tile of every bucket (LoweredStudy.plan())."
            )
        keeper = self.__dict__.get("_keeper")
        if keeper is not None:
            # The study it was cut from keeps the programs of one population plan, the one
            # prepared last: the programs kept for other widths are released before this study
            # compiles or evaluates anything (a study that still holds them keeps them alive).
            planned = {
                _batched.ProgramKey(
                    bucket.full_identity, mode, int(chunk.width), self._lanes.plan.mesh
                )
                for bucket in self._lanes.plan.buckets
                for mode, chunk in bucket.chunk.items()
            }
            for key in [key for key in keeper if key not in planned]:
                del keeper[key]
        executor = self.__dict__.get("_executor")
        if executor is not None:
            report = executor.prepare(
                self._lanes, cut=self.__dict__.get("_cut"), modes=modes, setup=self._setup
            )
        else:
            report = prepare_lanes(self._lanes, modes=modes, setup=self._setup)
        if keeper is not None:
            for program in self._lanes.programs.values():
                keeper[program.key] = program
        return report

    def evaluate(self, request: Request) -> Results:
        """Evaluate the experiments' solves for one request, retrying failed solves.

        Args:
            request: The `Request` naming the evaluation mode and the parameter points.

        Returns:
            The `Results` of every solve.
        """
        kinds, factors, _tolerance = self._retry
        executor = self.__dict__.get("_executor")
        if executor is not None:
            results = executor.evaluate(
                self._lanes,
                request,
                cut=self.__dict__.get("_cut"),
                retry=(kinds, factors),
                longest=self._study_ledger().longest(),
            )
        else:
            results = _batched.evaluate(
                self._lanes, request, retry_kinds=kinds, retry_first_step_factors=factors
            )
        self._ledger.record(results)
        study = self.__dict__.get("_fit_of")
        if study is not None:
            study.record(results)
        return results

    def _study_ledger(self) -> Ledger:
        """The ledger of every evaluation of the study, a fit's own included."""
        study = self.__dict__.get("_fit_of")
        return self._ledger if study is None else study

    def _for_fit(self) -> LoweredStudy:
        """This study with the evaluation state of one fit of its own.

        The returned study shares this study's lowered experiments, plan, lane
        numbers and programs (a program its ``prepare()`` compiles is this
        study's too), and owns what a series of evaluations learns: the step
        memory that orders the units of its calls, starting empty so that its
        first evaluation runs in canonical order, the inputs it keeps on the
        device, its lane route, and its ledger, which records its own
        evaluations only. Every evaluation it makes is recorded in this
        study's ledger as well, so this study's ledger keeps every lane's
        cost over all fits. A study that evaluates through no lanes of its
        own (one that stands in for a lowered study with its own
        ``evaluate``) has no such state to own and is copied as it is. In a
        run of several processes every process evaluates through its own
        lanes of the fit (the evaluation carries the fit), so this state is
        mirrored on every process.
        """
        fit = copy.copy(self)
        lanes = self.__dict__.get("_lanes")
        if lanes is not None:
            object.__setattr__(fit, "_lanes", _batched.with_own_state(lanes))
        ledger = self.__dict__.get("_ledger")
        if ledger is not None:
            object.__setattr__(fit, "_ledger", Ledger(ledger.lane_ids))
            object.__setattr__(fit, "_fit_of", self._study_ledger())
        object.__setattr__(fit, "_value_route", None)
        return fit

    def numeric_problem(self, optimizer: Any) -> Any:
        """The [LeastSquaresProblem][diff_soup.inference.LeastSquaresProblem] ``optimizer`` solves.

        As ``InferenceProblem.numeric_problem``, evaluated through a copy of
        this study with the evaluation state of one fit of its own
        (``_for_fit``): its value, gradient, exact Hessian and Gauss-Newton
        statistics are the lanes' where the plan offers their modes. When the
        study evaluates through lanes, the problem's structure holds
        ``"batched_lanes"``, ``problem.lanes`` is that copy and
        ``problem.joint_view()`` this study's joint problem.
        """
        fit = self._for_fit()
        lanes = fit if self.__dict__.get("_lanes") is not None else None
        return lower_estimation(self, fit, optimizer, lanes=lanes)

    def value_and_gradient(self, points: Any, *, weights: Any = None) -> ValueAndGradient:
        """Evaluate the objective and its gradient at many points in one batched request.

        Args:
            points: One point `(n,)` or `P` points `(P, n)` of the joint vector, or `(Q, P, n)` with `weights`.
            weights: A `(Q, m)` array with one weight per member of the first level for each group of points.

        Returns:
            A named tuple of `values` and `gradients`, `NaN` where a solve failed, with the `failures` and the raw `results`.
        """
        return self._value_and_gradient(points, weights, regularizer=True)

    def _value_and_gradient(
        self, points: Any, weights: Any, *, regularizer: bool
    ) -> ValueAndGradient:
        """``value_and_gradient``; with ``regularizer=False``, the lanes' part alone.

        Without the regularizer the value at a point is every lane's cost times
        its weight, summed, and the gradient is its gradient: the data's part of
        the objective, which ``log_likelihood`` reads.
        """
        route = self._route()
        n = route.n
        grid = np.asarray(points, dtype=route.dtype)
        if weights is None:
            grid = grid.reshape(1, n) if grid.shape == (n,) else grid
            if grid.ndim != 2 or grid.shape[1] != n:
                raise ValueError(
                    f"points= is (P, {n}), P points of the {n} coordinates, or one point ({n},); "
                    f"got the shape {np.shape(points)}."
                )
            grid = grid[None]
            routes = [route]
        else:
            rows = np.asarray(weights, dtype=route.dtype)
            nodes = int(route.tables.counts[0])
            if grid.ndim != 3 or grid.shape[2] != n or rows.shape != (grid.shape[0], nodes):
                raise ValueError(
                    f"with weights=, points= is (Q, P, {n}) and weights= is (Q, {nodes}), one "
                    f"weight per member of the level {route.names[0]!r} for every group of "
                    f"points; got points of shape {np.shape(points)} and weights of shape "
                    f"{np.shape(weights)}."
                )
            routes = [route.weighted((1, row), weight_priors=True) for row in rows]
        groups, count = grid.shape[:2]
        flat = grid.reshape(groups * count, n)
        results = self.evaluate(Request(mode="gradient", points=flat))
        lane_weights = np.repeat(np.stack([member.omega for member in routes]), count, axis=0)
        if results.levels is not None:
            values, gradients = self._summed_member_gradients(
                route, results, None if weights is None else rows, count, lane_weights
            )
        else:
            values, gradients = reduce_member_gradients(
                route.tables, results, results.lane_ids, lane_weights
            )
        if regularizer and int(np.size(route.view.regularization.residual(jnp.asarray(self.x0)))):
            for unit, point in enumerate(flat):
                member = routes[unit // count]
                values[unit] += member.penalty(point)
                gradients[unit] += member.penalty_gradient(point)
        failures = tuple(
            {
                "member": int(record["member"]) // count,
                "point": int(record["member"]) % count,
                "lane": record["lane"],
                "kind": record["kind"],
                "t_reached": record["t_reached"],
                "rung": record["rung"],
            }
            for record in results.failures()
            if lane_weights[int(record["member"]), route.position[record["lane"]]] != 0.0
        )
        shape = (count,) if weights is None else (groups, count)
        return ValueAndGradient(
            values.reshape(shape), gradients.reshape(shape + (n,)), failures, results
        )

    @staticmethod
    def _summed_member_gradients(
        route: Any, results: Results, rows: Any, count: int, lane_weights: np.ndarray
    ) -> tuple:
        """``reduce_member_gradients`` from results that carry the sums of the levels.

        Each unit's sums, every lane weighing one, are weighed by its group's
        weights of the members of level 1 after the sum; a unit with a failed
        lane of nonzero weight gets NaN.
        """
        kinds = np.asarray(results.status.kind)
        absent = (kinds == KINDS.index("absent")) & (lane_weights != 0.0)
        if np.any(absent):
            raise ValueError(
                "the objective sums every lane of nonzero weight, and these results left such "
                "lanes out; evaluate every lane of nonzero weight."
            )
        failed = np.any(np.asarray(results.status.failed) & (lane_weights != 0.0), axis=1)
        units = kinds.shape[0]
        values = np.zeros((units,), dtype=route.dtype)
        gradients = np.zeros((units, route.n), dtype=route.dtype)
        for unit in range(units):
            weights = None if rows is None else (1, rows[unit // count])
            sums = weigh_levels(route.tables, member_row(results.levels, unit), weights)
            values[unit], gradients[unit] = joint_gradient(route.tables, sums)
        values[failed] = np.nan
        gradients[failed] = np.nan
        return values, gradients

    def _route(self) -> Any:
        """The lane route of this study's objective, built on the first call and kept."""
        kept = self.__dict__.get("_value_route")
        route = kept[1] if kept is not None and kept[0]() is self else None
        if route is None:
            from diff_soup.inference.hierarchical.level_solvers import TrustRegion, _LaneRoute

            route = _LaneRoute(TrustRegion(), self)
            # The kept route refers to this study weakly: keeping it makes no reference cycle,
            # so the study's programs are still freed when the study is dropped.
            route.problem = weakref.proxy(self)
            object.__setattr__(self, "_value_route", (weakref.ref(self), route))
        return route

    def _offers(self, mode: str) -> bool:
        """Whether the lanes evaluate the whole objective in ``mode``.

        True when the plan refused no experiment and the program of every
        bucket runs in ``mode``.
        """
        if self._lanes.refused:
            return False
        return all(mode in entry.bucket.chunk for entry in self._lanes.buckets)

    def _ready(self, mode: str) -> None:
        """Compile the program of ``mode`` for every bucket that lacks it (``prepare()``).

        Called eagerly or while JAX traces a function that evaluates the lanes:
        the compile then runs on concrete numbers, outside that trace, before
        the traced program runs (a program cannot be compiled from inside the
        host callback of a running one).
        """
        programs = self._lanes.programs
        if any((entry.bucket.index, mode) not in programs for entry in self._lanes.buckets):
            with jax.core.eval_context():
                self.prepare(modes=(mode,))

    def _lanes_at(self, mode: str, x: Any) -> tuple:
        """The data's objective and its derivatives at one point, from one request of ``mode``.

        Every planned lane is evaluated at ``x``; each lane's cost, gradient and
        matrix (``J^T J`` in the ``blocks`` mode, the exact Hessian of its cost in
        the ``hessian`` mode) are added into the coordinates of ``x`` its narrow
        point reads. ``prepare()`` compiles ``mode`` first when a bucket has not.

        Returns:
            ``(cost, gradient, matrix, rows, failed, results)``: the gradient is
            ``None`` in the ``value`` mode and the matrix ``None`` outside the
            ``blocks`` and ``hessian`` modes; ``rows`` counts the lanes' residual
            rows and ``failed`` the lanes that failed, and then the cost and every
            entry are NaN; ``results`` are the request's.
        """
        self._ready(mode)
        point = np.asarray(x, dtype=np.asarray(self.x0).dtype).reshape(-1)
        results = self.evaluate(Request(mode=mode, points=point))
        if results.levels is not None:
            return self._summed_lanes_at(point, results)
        n = point.size
        gradient = None if results.gradient is None else np.zeros((n,), dtype=point.dtype)
        matrix = None if results.normal is None else np.zeros((n, n), dtype=point.dtype)
        for position, lane in enumerate(results.lane_ids):
            bucket, slot = self._lanes.where[lane]
            columns = self._lanes.buckets[bucket].gather[slot]
            if gradient is not None:
                gradient[columns] += np.asarray(results.gradient[0, position], dtype=point.dtype)
            if matrix is not None:
                block = np.asarray(results.normal[0, position], dtype=point.dtype)
                matrix[np.ix_(columns, columns)] += block
        failed = int(np.count_nonzero(np.asarray(results.status.kind)[0]))
        cost = math.nan if failed else float(np.sum(np.asarray(results.cost, dtype=point.dtype)[0]))
        for value in (gradient, matrix):
            if failed and value is not None:
                value[...] = np.nan
        rows = int(np.sum(np.asarray(results.rows)[0]))
        return cost, gradient, matrix, rows, failed, results

    def _summed_lanes_at(self, point: np.ndarray, results: Results) -> tuple:
        """``_lanes_at`` from results that carry the sums of the levels (lanes in several processes).

        The level tables hold every nonzero block of the lanes' matrix once, so
        the matrix and the gradient of the joint vector are laid out from the
        sums of the one member of ``results``.
        """
        tables = self._route().tables
        sums = member_row(results.levels, 0)
        _, gradient = joint_gradient(tables, sums)
        gradient = gradient.astype(point.dtype)
        cost = float(np.sum(np.asarray(results.cost, dtype=point.dtype)[0]))
        matrix = None
        if results.mode in ("blocks", "hessian"):
            matrix = joint_matrix(tables, sums).astype(point.dtype)
        kinds = np.asarray(results.status.kind)[0]
        failed = int(np.count_nonzero(np.asarray(results.status.failed)[0]))
        if np.any(kinds == KINDS.index("absent")):
            raise ValueError(
                "the objective at a point sums every lane, and these results left lanes out; "
                "evaluate every lane."
            )
        if failed:
            cost = math.nan
            gradient[...] = np.nan
            if matrix is not None:
                matrix[...] = np.nan
        rows = int(np.sum(np.asarray(results.rows)[0]))
        return cost, gradient, matrix, rows, failed, results

    def _prior_rows(self, point: Any) -> Any:
        """The rows of the ``MAP`` priors at ``point`` and their Jacobian; ``None`` without priors."""
        if not self._prior_blocks:
            return None
        from diff_soup.inference.solvers.tangent import value_and_jacfwd

        def rows(x):
            return jnp.concatenate(
                [prior.residual(x[columns]) for columns, prior in self._prior_blocks]
            )

        residual, jacobian = value_and_jacfwd(rows, jnp.asarray(point))
        return np.asarray(residual), np.asarray(jacobian)

    def log_likelihood(self, x: Any) -> Any:
        """The log-likelihood of the measurements at `x`, differentiable with JAX.

        Args:
            x: A point of the joint parameter vector.

        Returns:
            A scalar JAX array.
        """
        if self._log_normalizer is None or not (self._offers("value") and self._offers("gradient")):
            return super().log_likelihood(x)
        dtype = np.asarray(self.x0).dtype
        point = jnp.asarray(x, dtype=dtype)

        def costs(points):
            grid = np.asarray(points, dtype=dtype)
            return self._lane_costs(grid.reshape(-1, grid.shape[-1])).reshape(grid.shape[:-1])

        def costs_and_gradients(points):
            grid = np.asarray(points, dtype=dtype)
            out = self._value_and_gradient(
                grid.reshape(-1, grid.shape[-1]), None, regularizer=False
            )
            return out.values.reshape(grid.shape[:-1]), out.gradients.reshape(grid.shape)

        # A missing mode is compiled here, eagerly or while JAX traces, and never from the
        # host functions, which may run inside the callback of a running program.
        @jax.custom_jvp
        def cost(at):
            self._ready("value")
            return _on_the_host(costs, at, jax.ShapeDtypeStruct((), dtype))

        @cost.defjvp
        def cost_jvp(primals, tangents):
            (at,), (tangent,) = primals, tangents
            self._ready("gradient")
            shapes = (jax.ShapeDtypeStruct((), dtype), jax.ShapeDtypeStruct(at.shape, dtype))
            value, gradient = _on_the_host(costs_and_gradients, at, shapes)
            return value, jnp.sum(gradient * tangent)

        return self._log_normalizer - cost(point)

    def _lane_costs(self, points: Any) -> np.ndarray:
        """The lanes' costs summed at every point of ``points`` ``(P, n)``, from one ``value`` request.

        NaN at a point where a lane failed.
        """
        dtype = np.asarray(self.x0).dtype
        results = self.evaluate(Request(mode="value", points=np.asarray(points, dtype=dtype)))
        costs = np.sum(np.asarray(results.cost, dtype=dtype), axis=1)
        costs[np.any(np.asarray(results.status.kind) != 0, axis=1)] = np.nan
        return costs

    @property
    def loss_fn(self):
        """The objective of the data and the priors at one point, evaluated by the lanes.

        ``x -> cost``: one request of the ``value`` mode, the lanes' costs
        summed, plus half the sum of the squared rows of the priors of a
        ``MAP`` objective; NaN where a lane failed. A host function, which JAX
        does not trace. When the plan refused an experiment, as
        ``InferenceProblem.loss_fn``.
        """
        if not self._offers("value"):
            return super().loss_fn
        return self._lane_loss

    def _lane_loss(self, x: Any) -> float:
        cost = self._lanes_at("value", x)[0]
        return cost + float(self._prior_objective(jnp.asarray(x)))

    def loss_and_grad_fn(self, ad: str = "reverse"):
        """The objective of the data and the priors and its gradient, evaluated by the lanes.

        ``x -> (cost, gradient)``: one request of the ``gradient`` mode, every
        lane's gradient added into the coordinates of ``x``, plus the priors';
        NaN where a lane failed. A bucket's gradient mode differentiates as its
        solver does (in reverse mode when the solver carries a reverse adjoint,
        from the forward Jacobian otherwise), whichever of ``"forward"`` and
        ``"reverse"`` ``ad`` names. A host function, which JAX does not trace.
        When the plan refused an experiment, as
        ``InferenceProblem.loss_and_grad_fn``.

        Raises:
            ConfigurationError: If ``ad`` is neither ``"forward"`` nor
                ``"reverse"``.
        """
        if not self._offers("gradient"):
            return super().loss_and_grad_fn(ad)
        if ad not in ("forward", "reverse"):
            raise ConfigurationError(
                f"optsolve differentiation mode must be 'forward' or 'reverse'; got {ad!r}."
            )
        return self._lane_loss_and_gradient

    def _lane_loss_and_gradient(self, x: Any) -> tuple:
        cost, gradient, _matrix, _rows, _failed, _results = self._lanes_at("gradient", x)
        prior, prior_gradient = jax.value_and_grad(self._prior_objective)(jnp.asarray(x))
        return cost + float(prior), jnp.asarray(gradient + np.asarray(prior_gradient))

    def loss_grad_hessian_fn(self, composition: str = "auto"):
        """The objective of the data and the priors, its gradient and its exact Hessian.

        When ``composition`` resolves to ``"forward-over-reverse"`` (what
        ``"auto"`` resolves to when the experiments' adjoints offer it) and
        every bucket runs the ``hessian`` mode, the lanes evaluate it:
        ``x -> (cost, gradient, Hessian)`` from one request of that mode, every
        lane's Hessian and gradient added into the coordinates of ``x``, plus
        the priors'; NaN where a lane failed. A host function, which JAX does
        not trace. Another composition, or a plan that refused an experiment,
        is evaluated as ``InferenceProblem`` evaluates it.
        """
        resolved = self.resolved_hessian_composition(composition)
        if resolved != "forward-over-reverse" or not self._offers("hessian"):
            return super().loss_grad_hessian_fn(composition)
        return self._lane_loss_gradient_hessian

    def _lane_loss_gradient_hessian(self, x: Any) -> tuple:
        cost, gradient, hessian, _rows, _failed, _results = self._lanes_at("hessian", x)
        point = jnp.asarray(x)
        prior, prior_gradient = jax.value_and_grad(self._prior_objective)(point)
        prior_hessian = jax.hessian(self._prior_objective)(point)
        return (
            cost + float(prior),
            jnp.asarray(gradient + np.asarray(prior_gradient)),
            jnp.asarray(hessian + np.asarray(prior_hessian)),
        )

    @property
    def gauss_newton_fn(self):
        """The objective of the data and the priors, its gradient and its Gauss-Newton matrix.

        ``x -> (cost, gradient, matrix, failed)`` from one request of the
        ``blocks`` mode: every lane's ``J^T r`` and ``J^T J`` added into the
        coordinates of ``x``, and the rows of the priors of a ``MAP`` objective
        added as rows (``J_p^T r_p``, ``J_p^T J_p``, half their squared sum);
        ``failed`` counts the lanes that failed, and then the numbers are NaN.
        ``LevenbergMarquardt`` and ``NewtonTrustRegion(hessian="gauss-newton")``
        fit a problem that offers it from it. ``None`` when a bucket's solver
        carries a reverse adjoint, which has no ``blocks`` mode, or the plan
        refused an experiment.
        """
        return self._lane_gauss_newton if self._offers("blocks") else None

    def _lane_gauss_newton(self, x: Any) -> tuple:
        cost, gradient, matrix, _rows, failed, _results = self._lanes_at("blocks", x)
        prior = self._prior_rows(x)
        if prior is not None:
            residual, jacobian = prior
            cost += 0.5 * float(residual @ residual)
            gradient = gradient + jacobian.T @ residual
            matrix = matrix + jacobian.T @ jacobian
        return cost, jnp.asarray(gradient), jnp.asarray(matrix), failed

    def observed_information(self, x: Any) -> Any:
        """The observed information at `x`, the Hessian of the negative log-likelihood.

        Args:
            x: A point of the joint parameter vector.
        """
        if self.resolved_hessian_composition("auto") != "forward-over-reverse" or not self._offers(
            "hessian"
        ):
            return super().observed_information(x)
        return jnp.asarray(self._lanes_at("hessian", x)[2])

    def expected_information(self, x: Any) -> Any:
        """The expected information at `x`, the Gauss-Newton matrix, or `None` when the study does not provide it.

        Args:
            x: A point of the joint parameter vector.
        """
        if not self._offers("blocks"):
            return None
        return jnp.asarray(self._lanes_at("blocks", x)[2])

    def _experiment_scores(self, x: Any) -> Any:
        """The data's ``J^T J`` and every experiment's ``J_e^T r_e`` at ``x``, from the lanes.

        One request of the ``blocks`` mode. Returns ``(bread, scores, rows)``:
        ``(n, n)``; ``(E, n)``, one row per experiment in the study's order, the
        lanes of an experiment added together; and the residual rows of every
        experiment. NaN where a lane failed. ``None`` when a bucket's solver
        carries a reverse adjoint, which has no ``blocks`` mode, or the plan
        refused an experiment.
        """
        if not self._offers("blocks"):
            return None
        if self.processes > 1:
            raise ValueError(
                "every experiment's score is the sum of its lanes' gradients, and the lanes of "
                f"this study run in {self.processes} processes, which return the sums of the "
                "parameter tree's levels only (the score of each experiment on the shared "
                "coordinates is not among them); compute cluster scores in a run of one process."
            )
        from diff_soup.inference.engine._layout import lane_layout

        _cost, _gradient, bread, _rows, failed, results = self._lanes_at("blocks", x)
        # A bucket names each lane by the study's ids and gives its index in the lane layout.
        experiments = lane_layout(self._inference_problem).experiments
        experiment_of = {
            lane: experiments[child]
            for bucket in self.plan().buckets
            for lane, child in zip(bucket.lanes, bucket.children)
        }
        n = bread.shape[0]
        scores = np.zeros((len(self.study.experiment_ids), n), dtype=bread.dtype)
        rows = np.zeros((len(self.study.experiment_ids),), dtype=np.int64)
        for position, lane in enumerate(results.lane_ids):
            bucket, slot = self._lanes.where[lane]
            columns = self._lanes.buckets[bucket].gather[slot]
            experiment = int(experiment_of[lane])
            scores[experiment, columns] += np.asarray(results.gradient[0, position])
            rows[experiment] += int(np.asarray(results.rows)[0, position])
        if failed:
            scores[...] = np.nan
        return bread, scores, rows

    def tables(self, lane_ids: Any) -> TreeTables | LevelTables:
        """The tables that map the given solves to their columns of the joint vector.

        Args:
            lane_ids: The ids of the solves, as a request's results list them.

        Returns:
            `LevelTables` for a `Hierarchy` study, `TreeTables` otherwise.
        """
        if isinstance(self.study.parameters, Hierarchy):
            return _batched.level_tables_of(
                self._lanes, lane_ids, self.study.parameters, self.study.experiments
            )
        return _batched.tables(self._lanes, lane_ids)


def prepare_windows(experiment: Experiment, trajectory: Any = None, **problem_kwargs: Any) -> Any:
    """Compute and store the window starts of a multiple-shooting experiment before a fit.

    A later fit of the same experiment loads them instead of solving again. Set
    `DIFFSOUP_CACHE_DIR` to a shared directory for a run of several processes.

    Args:
        experiment: The experiment whose windows to initialize.
        trajectory: A `Multishoot` applied to the experiment, or `None` to keep its own.
        **problem_kwargs: Further keywords of `InferenceProblem`, such as `parameters=`.

    Returns:
        The key the windows are stored under, or `None` when there is nothing to store.
    """
    problem = InferenceProblem(experiment, trajectory=trajectory, **problem_kwargs)
    leaf = problem._inference_problem
    leaf.persist_window_initialization()
    return getattr(leaf, "_window_content_key", None)
