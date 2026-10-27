"""Trust-region fits of a lowered study on its lanes, along the tree of its parameters."""

from __future__ import annotations

import copy
import dataclasses
import time
from types import MappingProxyType
from typing import Any, ClassVar, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._distributed import any_process, assert_decision, is_primary
from diff_soup._errors import ConfigurationError
from diff_soup.inference._abc import RESUME, OptSolver
from diff_soup.inference.engine._layout import lane_layout
from diff_soup.inference.engine._levels import (
    LevelTables,
    _lane_weights,
    member_row,
    reduce_levels,
    weigh_levels,
)
from diff_soup.inference.engine._reducers import refuse_failed
from diff_soup.inference.engine._request import Request
from diff_soup.inference.engine._status import LaneStatus
from diff_soup.inference.fitting._trial import trial_is_valid
from diff_soup.inference.fitting.bounds import resolve_decision_bounds
from diff_soup.inference.fitting.callbacks import StopFit
from diff_soup.inference.fitting.result import (
    EvalCounts,
    OptResult,
    StepInfo,
    TerminationStatus,
    TrustRegionReport,
)
from diff_soup.inference.fitting.scales import ConstantScale, HessianScale, JacScale
from diff_soup.inference.hierarchical import _bounds as box
from diff_soup.inference.hierarchical import _level_elimination as elimination
from diff_soup.inference.hierarchical._constraints import _constraint_rows
from diff_soup.inference.solvers._loop import (
    _NOISE_RUN,
    _TINY,
    LoopState,
    _after_attempt,
    _floor_level,
    _floor_parameter_radius,
    _floor_radius,
    _floor_sigma,
    _gain_ratio,
    _judge,
    _jump_level,
    _median,
    _next_radius,
    _noise_dominated,
    _saved_count,
    _saved_value,
    _stop_after,
    cost_scale,
    cost_small,
    cost_threshold,
    loop_tolerances,
    step_small,
)
from diff_soup.inference.solvers._termination import step_budget

_LABEL = "TrustRegion"


def _lowered(solver: Any, problem: Any) -> Any:
    """``problem`` as the optimization problem of a lowered study's lanes.

    An optimization problem is returned as it is when its structure holds
    ``"batched_lanes"``; a lowered study is lowered for ``solver``.
    """
    structure = getattr(problem, "structure", None)
    if structure is None:
        lower = getattr(problem, "numeric_problem", None)
        if not callable(lower) or not callable(getattr(problem, "_offers", None)):
            raise TypeError(
                "TrustRegion fits the lanes of a lowered study, ds.lower(study); got "
                f"{type(problem).__name__}."
            )
        problem = lower(solver)
        structure = problem.structure
    if "batched_lanes" not in structure:
        raise ConfigurationError(
            "TrustRegion fits the lanes of a lowered study, and this problem provides "
            f"{sorted(structure)} without 'batched_lanes'; lower the study with "
            "ds.lower(study) and pass that."
        )
    return problem


def _quoted(names: Any) -> str:
    return ", ".join(repr(name) for name in names)


def _listed(values: Any, limit: int = 8) -> str:
    values = tuple(values)
    text = ", ".join(str(value) for value in values[:limit])
    if len(values) > limit:
        text += f" and {len(values) - limit} more"
    return text


def _strings(values: Any) -> np.ndarray:
    """Labels as a host array of strings (no object array)."""
    return np.asarray([str(value) for value in values], dtype=str)


def _string_rows(rows: Any, width: int) -> np.ndarray:
    """Rows of labels as a ``(len(rows), width)`` host array of strings."""
    rows = [[str(value) for value in row] for row in rows]
    return np.asarray(rows, dtype=str).reshape(len(rows), width)


def _padded_rows(rows: Any, fill: Any, dtype: Any) -> np.ndarray:
    """Rows of different lengths as one host array, each row padded with ``fill``."""
    rows = [list(row) for row in rows]
    width = max((len(row) for row in rows), default=0)
    padded = [row + [fill] * (width - len(row)) for row in rows]
    return np.asarray(padded, dtype=dtype).reshape(len(rows), width)


def _unpadded(row: Any) -> tuple:
    """A row of rungs written by ``_padded_rows`` with the fill -1, without its padding."""
    return tuple(int(value) for value in np.ravel(row) if int(value) >= 0)


NONFINITE_BLOCKS = "nonfinite_blocks_at_finite_cost"
"""The failure kind of a lane whose solve reached a finite cost with non-finite Gauss-Newton blocks."""


class _Failure(NamedTuple):
    """One failed lane: its id, kind, the time its solve reached, its member at every level.

    ``rungs`` holds the rungs its solve was tried at in that evaluation, in
    order. ``ignored`` marks a lane of weight zero, outside the objective: its
    failure is logged and decides nothing.
    """

    lane: str
    kind: str
    t_reached: float
    nodes: tuple
    rungs: tuple = ()
    ignored: bool = False


def _rungs_text(rungs: Any) -> str:
    """``after rung 0`` or ``after rungs 0, 1, 2``; empty when the rungs are not known."""
    rungs = tuple(rungs)
    if not rungs:
        return ""
    label = "rung" if len(rungs) == 1 else "rungs"
    return f", after {label} {', '.join(str(rung) for rung in rungs)}"


def _failures_text(failures: Any) -> str:
    return _listed(
        f"{failure.lane}: {failure.kind} at t = {failure.t_reached:.6g}{_rungs_text(failure.rungs)}"
        for failure in failures
    )


# ---------------------------------------------------------------------------
# The lanes of a lowered study
# ---------------------------------------------------------------------------


def _level_weights(weights: Any) -> Any:
    """``weights=`` as ``None`` or ``(level, values)``; a 1-D array weighs level 1."""
    if weights is None:
        return None
    if (
        isinstance(weights, (tuple, list))
        and len(weights) == 2
        and (isinstance(weights[0], str) or np.ndim(weights[0]) == 0)
        and np.ndim(weights[1]) == 1
    ):
        return (weights[0], weights[1])
    return (1, weights)


def _lane_positions(plan: Any, ids: tuple) -> np.ndarray:
    """Per lane of ``ids``, its index in the study's lane layout, as the plan's buckets give it."""
    index = {
        lane: child for bucket in plan.buckets for lane, child in zip(bucket.lanes, bucket.children)
    }
    return np.asarray([index[lane] for lane in ids], dtype=np.int64)


def _experiment_level(problem: Any, ids: tuple, plan: Any) -> tuple:
    """The experiment level of a ``ParameterSet`` study: experiment indices, lane tables, width.

    Returns the study's experiment ids, each lane's experiment index, the
    level's ``(lanes, present)`` tables (one node per experiment, holding the
    lanes of its windows in time order when it is fitted in windows) and the
    lanes' one narrow width.
    """
    experiments = tuple(problem.study.experiment_ids)
    layout = lane_layout(problem.diagnostic_problem)
    lane_experiments = np.asarray(layout.experiments, dtype=np.int64)[_lane_positions(plan, ids)]
    widths = {int(bucket.width) for bucket in plan.buckets}
    if len(widths) != 1:
        problem.tables(ids)  # names the lanes of each width
        raise ValueError(f"the lanes have the narrow widths {sorted(widths)}; give them one width.")
    most = max(int(np.bincount(lane_experiments, minlength=len(experiments)).max()), 1)
    lanes = np.zeros((len(experiments), most), dtype=np.int32)
    present = np.zeros((len(experiments), most), dtype=bool)
    for position, experiment in enumerate(lane_experiments):
        slot = int(np.count_nonzero(present[experiment]))
        lanes[experiment, slot] = position
        present[experiment, slot] = True
    return experiments, lane_experiments, lanes, present, widths.pop()


def _experiment_level_tables(problem: Any, ids: tuple, plan: Any, device: Any) -> LevelTables:
    """The depth-1 level tables of a ``ParameterSet`` study without an owner tier, on ``device``.

    Level 1 is the experiments, of narrow width 0: every lane reads the shared
    coordinates only.
    """
    experiments, lane_experiments, lanes, present, width = _experiment_level(problem, ids, plan)
    return LevelTables(
        nodes=lane_experiments[:, None],
        counts=(len(experiments),),
        widths=(width, 0),
        parents=(None, None),
        lanes=(None, lanes),
        present=(None, present),
        names=("experiment",),
        lane_ids=ids,
        device=device,
    )


def _owner_level_tables(problem: Any, view: Any, ids: tuple, plan: Any, device: Any) -> LevelTables:
    """The depth-2 level tables of a study declared with a ``ParameterSet``, on ``device``.

    Level 1 is the owner tier, level 2 the experiments; a lane's nodes are its
    experiment's owner and the experiment itself.
    """
    layout = lane_layout(view)
    owners = np.asarray(layout.nodes, dtype=np.int64)[_lane_positions(plan, ids)]
    n_owners = len(layout.labels)
    experiments, lane_experiments, lanes_two, present_two, width = _experiment_level(
        problem, ids, plan
    )
    # The owner of every experiment: the node of its lanes.
    parents = np.zeros((len(experiments),), dtype=np.int64)
    parents[np.asarray(layout.experiments, dtype=np.int64)] = layout.nodes
    shared = layout.shared
    owner_width = layout.tier_width
    per_owner = [np.flatnonzero(owners == owner) for owner in range(n_owners)]
    most = max(max((len(entry) for entry in per_owner), default=1), 1)
    lanes_one = np.zeros((n_owners, most), dtype=np.int32)
    present_one = np.zeros((n_owners, most), dtype=bool)
    for owner, entry in enumerate(per_owner):
        lanes_one[owner, : len(entry)] = entry
        present_one[owner, : len(entry)] = True
    return LevelTables(
        nodes=np.stack([owners, lane_experiments], axis=1),
        counts=(n_owners, len(experiments)),
        widths=(shared, owner_width, width - shared - owner_width),
        parents=(None, None, parents),
        lanes=(None, lanes_one, lanes_two),
        present=(None, present_one, present_two),
        names=(str(problem.study._tied_clusters()[0]), "experiment"),
        lane_ids=ids,
        device=device,
    )


def _route_tables(problem: Any, view: Any, ids: tuple, plan: Any, device: Any) -> tuple:
    """``(tables, members)``: the level tables of the lanes ``ids`` and the labels of every level.

    A [Hierarchy][diff_soup.inference.Hierarchy] study takes the tables of its
    hierarchy; a ``ParameterSet`` study with an owner tier the owners and the
    experiments; any other study one level of experiments. ``ids`` may be any
    subset of the planned lanes, in canonical order: the tables keep every
    node of the study, so the sums over disjoint subsets add up.
    """
    from diff_soup.inference.statistics.hierarchy import Hierarchy

    study = problem.study
    if isinstance(study.parameters, Hierarchy):
        tables = problem.tables(ids)
        members = tuple(
            tuple(dict.fromkeys(study.clusters[name])) for name in tables.names[:-1]
        ) + (tuple(study.experiment_ids),)
    elif getattr(view, "owner_ids", ()):
        tables = _owner_level_tables(problem, view, ids, plan, device)
        members = (tuple(view.owner_ids), tuple(study.experiment_ids))
    else:
        # One experiment, or experiments that share every coordinate: one level.
        tables = _experiment_level_tables(problem, ids, plan, device)
        members = (tuple(study.experiment_ids),)
    return tables, members


class _LaneRoute:
    """What a fit on the lanes of a lowered study needs: tables, weights, rows, bounds, labels.

    ``block_weights`` is the one definition of the regularizer's weights: one
    weight per block of ``view.regularization.block_rows``, in its order, or
    ``None`` when every block weighs one. The level system, the trial cost, the
    nested mode's member costs and ``LoweredStudy.value_and_gradient`` all read
    it.

    ``device`` is the JAX device of the lowered study's plan, read once: the
    level tables, the lane results the route reduces, its level systems and the
    points its regularizer is evaluated at are placed there. ``None`` (a plan
    that names no device) places them on the default device.
    """

    block_weights: Any = None
    device: Any = None
    # The gradient and diagonal of the last system, and the regularizer's blocks at the
    # last point it was evaluated at, each kept for the next request of the same.
    kept_vectors: Any = None
    kept_regularizer: Any = None
    # The index tables of every level system of the fit, built and placed once.
    layout: Any = None

    def __init__(self, solver: Any, problem: Any):
        lowered = _lowered(solver, problem)
        view = lowered.joint_view()
        # The fit evaluates through the copy of the study the lowering made for it, with
        # its own evaluation state: its step memory (the order of the units in the calls),
        # the inputs it keeps on the device and its ledger start empty, so its first
        # evaluation runs in canonical order and nothing of another fit of the same
        # lowered study reaches it.
        problem = lowered.lanes
        n = int(view.n_params)
        fixed = tuple(sorted({int(index) for index in solver.fixed_indices}))
        if any(index < 0 or index >= n for index in fixed):
            raise ConfigurationError("TrustRegion fixed_indices must name valid joint coordinates.")
        plan = problem.plan()
        device = getattr(plan, "device", None)
        # Every planned lane in canonical order: by its index in the study's lane layout.
        ids = tuple(
            lane
            for _, lane in sorted(
                (child, lane)
                for bucket in plan.buckets
                for child, lane in zip(bucket.children, bucket.lanes)
            )
        )
        tables, members = _route_tables(problem, view, ids, plan, device)
        counts, widths, names = tables.counts, tables.widths, tables.names
        fitted = int(widths[0]) + sum(
            int(count) * int(width) for count, width in zip(counts, widths[1:])
        )
        if fitted != n:
            levels = ", ".join(
                f"{name} {count} x {width}" for name, count, width in zip(names, counts, widths[1:])
            )
            raise ConfigurationError(
                f"the lanes of this study fit {fitted} coordinates level by level (shared "
                f"{widths[0]}, levels {levels}) and its parameter vector has {n}; coordinates "
                "outside the levels, such as initial states, are not fitted on the lanes."
            )
        if solver.nested_at is not None and solver.nested_at not in names:
            raise ConfigurationError(
                f"TrustRegion nested_at={solver.nested_at!r} is not a level of this study; its "
                f"levels are {_quoted(names)}."
            )
        dtype = np.asarray(problem.x0).dtype
        self.device = device
        self.problem = problem
        self.layout = elimination.level_layout(tables, device)
        self.view = view
        self.n = n
        self.fixed = fixed
        self.ids = ids
        self.tables = tables
        self.names = names
        self.members = members
        self.depth = len(counts)
        self.nodes = np.asarray(tables.nodes)
        self.dtype = dtype
        self.constraints = solver.constraints
        self.gauges = solver.gauges
        # The member each regularizer block belongs to, per weighted level; computed once.
        self.attributions: dict = {}
        self.weigh_by(solver.weights, weight_priors=solver.weight_priors)
        bounds = resolve_decision_bounds(view)
        self.bounds = (
            None
            if bounds is None
            else (np.asarray(bounds[0], dtype=dtype), np.asarray(bounds[1], dtype=dtype))
        )
        self.fingerprint = str(problem.parameter_schema.fingerprint)
        self.position = {lane: index for index, lane in enumerate(ids)}
        # The ceiling of the noise of the cost's evaluation, or None when it is unknown.
        self.evaluation_noise = problem.evaluation_noise

    def weigh_by(self, weights: Any, *, weight_priors: bool) -> None:
        """Set the route's weights: the lanes', the regularizer blocks' and the gauges'.

        ``weights`` is ``None`` or ``(level, values)`` (a 1-D array weighs
        level 1). Every lane takes its member's weight (``omega``); with
        ``weight_priors`` every regularizer block takes the weight of the
        member of the weighted level it belongs to, a block above that level
        weighs one (``block_weights``, ``None`` when every block weighs one);
        a gauge across the members of level 1 takes their weights.
        """
        tables, names, dtype = self.tables, self.names, self.dtype
        weights = _level_weights(weights)
        omega = _lane_weights(tables, weights, len(self.ids), dtype)
        level_one = np.ones((int(tables.counts[0]),), dtype=dtype)
        block_weights = None
        if weights is not None:
            level = weights[0]
            number = names.index(level) + 1 if isinstance(level, str) else int(level)
            if number == 1:
                level_one = np.asarray(weights[1], dtype=dtype)
            if weight_priors:
                # Every regularizer block takes the weight of the member of the weighted
                # level it belongs to (its deepest node or that node's ancestor); a block
                # above that level weighs one.
                if number not in self.attributions:
                    self.attributions[number] = elimination.regularizer_block_members(
                        self.view.regularization, self.problem.x0, tables, number, self.device
                    )
                attributed = self.attributions[number]
                if attributed is not None:
                    values = np.asarray(weights[1], dtype=np.float64)
                    per_block = np.where(attributed >= 0, values[np.maximum(attributed, 0)], 1.0)
                    if np.any(per_block != 1.0):
                        block_weights = tuple(float(value) for value in per_block)
        matrix, rhs = _constraint_rows(
            self.view, self.constraints, self.gauges, jax.device_put(level_one, self.device), dtype
        )
        self.weights = weights
        self.omega = omega
        self.block_weights = block_weights
        self.constraint_matrix = np.asarray(matrix, dtype=dtype)
        self.constraint_rhs = np.asarray(rhs, dtype=dtype)

    def weighted(self, weights: Any, *, weight_priors: bool) -> _LaneRoute:
        """A copy of this route that weighs by ``weights``; this route is unchanged.

        The copy's lane weights, block weights and gauge rows come from
        [weigh_by][diff_soup.inference.hierarchical.level_solvers._LaneRoute.weigh_by],
        the one place that defines them; everything else is this route's.
        """
        other = copy.copy(self)
        other.weigh_by(weights, weight_priors=weight_priors)
        return other

    def refuse_inner_constraints(self, level_name: str) -> None:
        """Refuse equality rows that address coordinates at or below the cut level."""
        cut = self.names.index(level_name) + 1
        widths, counts = self.tables.widths, self.tables.counts
        prefix = int(widths[0]) + sum(
            int(counts[level - 1]) * int(widths[level]) for level in range(1, cut)
        )
        group_of = {
            int(column): str(name)
            for name, columns in zip(self.view.param_group_names, self.view.param_group_cols)
            for column in columns
        }
        for row_index, row in enumerate(self.constraint_matrix):
            below = np.flatnonzero(row[prefix:] != 0.0)
            if below.size:
                column = int(below[0]) + prefix
                raise ConfigurationError(
                    f"with nested_at={level_name!r} a constraint may address coordinates above "
                    f"that level only; constraint row {row_index} addresses "
                    f"{group_of.get(column, f'column {column}')}."
                )

    def evaluate(
        self,
        mode: str,
        point: np.ndarray,
        *,
        lanes: Any = None,
        rungs: Any = None,
        first_steps: Any = None,
    ) -> Any:
        """One request at one point: every planned lane, or the planned lanes ``lanes``.

        ``rungs`` and ``first_steps`` are the request's (one row, one column per
        requested lane): the rung each lane runs at and the first step of its
        rung-0 solve; ``None`` runs every lane at rung 0.
        """
        results = self.problem.evaluate(
            Request(mode=mode, points=point, lanes=lanes, rungs=rungs, first_steps=first_steps)
        )
        if lanes is None:
            if tuple(results.lane_ids) != self.ids:
                raise RuntimeError(
                    f"the lowered study returned the lanes {_listed(results.lane_ids)} for a "
                    f"request of every planned lane, {_listed(self.ids)}."
                )
            return results
        asked = {str(lane) for lane in lanes}
        expected = tuple(lane for lane in self.ids if lane in asked)
        if tuple(results.lane_ids) != expected:
            raise RuntimeError(
                f"the lowered study returned the lanes {_listed(results.lane_ids)} for a request "
                f"of the planned lanes {_listed(expected)}."
            )
        return results

    def failures(self, results: Any) -> tuple:
        """The failed lanes of ``results``, each with its member at every level and its rungs.

        The rungs tried are those of the lane's record in ``results.retries``
        when the evaluation solved it again, else the one rung it ran at.
        """
        tried = {
            (int(record["member"]), str(record["lane"])): tuple(int(r) for r in record["rungs"])
            for record in getattr(results, "retries", ())
        }
        records = []
        for failure in results.failures():
            lane = str(failure["lane"])
            row = self.nodes[self.position[lane]]
            nodes = tuple(str(self.members[level][int(row[level])]) for level in range(self.depth))
            rungs = tried.get((int(failure.get("member", 0)), lane), (int(failure.get("rung", 0)),))
            records.append(
                _Failure(lane, str(failure["kind"]), float(failure["t_reached"]), nodes, rungs)
            )
        return tuple(records)

    def weigh(self, failures: tuple) -> tuple:
        """``(counted, ignored)``: the failed lanes of nonzero weight, and those of weight zero.

        A lane of weight zero is outside the objective: its failure is logged
        as ignored and never makes a trial invalid or an evaluation an error.
        """
        counted, ignored = [], []
        for failure in failures:
            if float(self.omega[self.position[failure.lane]]) == 0.0:
                ignored.append(failure._replace(ignored=True))
            else:
                counted.append(failure)
        return tuple(counted), tuple(ignored)

    def system(self, results: Any, point: np.ndarray, *, member: int = 0) -> Any:
        """The level system at ``point`` from a ``blocks`` (or ``hessian``) evaluation there.

        ``member`` is the population member of ``results`` whose lanes are
        reduced; a fit's own requests hold the one member 0. Results of lanes
        evaluated in several processes carry the sums of the levels already
        (``Results.levels``, every lane weighing one); the route's weights are
        then applied to those sums.
        """
        if getattr(results, "levels", None) is not None:
            blocks = self._summed_levels(results, member)
        else:
            # Without a device, the one-argument call that results defining their own
            # ``lane_result(member)`` take.
            lanes = (
                results.lane_result(member)
                if self.device is None
                else results.lane_result(member, self.device)
            )
            blocks = reduce_levels(self.tables, lanes, results.lane_ids, self.weights)
        dtype = np.dtype(blocks.shared.dtype)
        return elimination.level_system(
            blocks,
            self.tables,
            point,
            constraint_matrix=self.constraint_matrix,
            constraint_rhs=self.constraint_rhs,
            fixed_indices=self.fixed,
            regularizer=self.view.regularization,
            lane_weights=self.omega,
            members=self.members,
            block_weights=self.block_weights,
            device=self.device,
            layout=self.levels(),
            regularizer_blocks=self.regularizer_at(point, dtype),
        )

    def levels(self) -> Any:
        """The [LevelLayout][diff_soup.inference.hierarchical._level_elimination.LevelLayout] of the tables, built once."""
        if self.layout is None:
            self.layout = elimination.level_layout(self.tables, self.device)
        return self.layout

    def regularizer_at(self, point: np.ndarray, dtype: Any = None) -> Any:
        """The regularizer's blocks, cost and touched nodes at ``point`` under the route's weights.

        ``elimination.regularizer_blocks_at``'s; the blocks of the last point
        asked for are kept, so the system at an accepted trial reuses the
        evaluation its trial cost made. ``None`` without a regularizer.
        """
        regularizer = self.view.regularization
        if regularizer is None:
            return None
        dtype = np.dtype(self.dtype if dtype is None else dtype)
        point = np.asarray(point, dtype=dtype)
        key = (dtype.str, point.tobytes(), self.block_weights)
        kept = self.kept_regularizer
        if kept is None or kept[0] != key:
            blocks = elimination.regularizer_blocks_at(
                regularizer,
                point,
                self.levels(),
                dtype,
                getattr(self.tables, "names", None),
                self.members,
                self.block_weights,
            )
            kept = self.kept_regularizer = (key, blocks)
        return kept[1]

    def vectors(self, system: Any) -> tuple:
        """``elimination.level_vectors(system)``, kept for the last system asked for.

        The attempts of a fit that reject their trials step from the same
        system; its gradient and diagonal move to the host once. Copies are
        returned, so a caller may change them.
        """
        kept = self.kept_vectors
        if kept is None or kept[0] is not system:
            kept = self.kept_vectors = (system, *elimination.level_vectors(system))
        return kept[1].copy(), kept[2].copy()

    def solve_attempt(self, system: Any, coordinates: Any, radius: float) -> Any:
        """The solve at zero damping, its predicted decrease and the step of ``radius``, at once.

        ``elimination.attempt_solve`` in the subproblem's ``coordinates``: one
        program on the device and one transfer, in place of the chain of
        ``solve_scaled``, the full step's decrease and ``level_secular_step``.
        """
        return elimination.attempt_solve(system, coordinates.scale, radius, coordinates.addition)

    def _summed_levels(self, results: Any, member: int) -> Any:
        """The level blocks of ``member`` from results that carry the sums of the levels.

        The lanes' failures and unevaluated units are refused under the
        route's weights as ``reduce_levels`` refuses them; the sums, every
        lane weighing one, are then weighed by the route's weights.
        """
        rows = np.flatnonzero(np.asarray(results.members) == int(member))
        if rows.shape[0] != 1:
            raise ValueError(
                f"member {int(member)} is not among the members of these results "
                f"{np.asarray(results.members).tolist()}; ask for one of them."
            )
        row = int(rows[0])
        if tuple(results.lane_ids) != tuple(self.ids):
            raise ValueError(
                f"the results are for the lanes {_listed(results.lane_ids)} and the route's tables "
                f"for {_listed(self.ids)}; a level system sums every planned lane."
            )
        status = LaneStatus(
            kind=np.asarray(results.status.kind)[row],
            t_reached=np.asarray(results.status.t_reached)[row],
            rung=np.asarray(results.status.rung)[row],
        )
        refuse_failed(status, tuple(results.lane_ids), None if self.weights is None else self.omega)
        sums = weigh_levels(self.tables, member_row(results.levels, row), self.weights)
        return jax.device_put(sums, self.device)

    def penalty(self, point: np.ndarray) -> float:
        """The regularizer's cost at ``point``: every block's ``0.5 |r_b|^2`` times its weight."""
        regularizer = self.view.regularization
        if self.block_weights is None:
            return float(regularizer.penalty(jax.device_put(point, self.device)))
        declared = tuple(regularizer.block_rows(jax.device_put(point, self.device)))
        if len(declared) != len(self.block_weights):
            raise ValueError(
                f"the regularizer's block_rows(x) declares {len(declared)} blocks at this point "
                f"and {len(self.block_weights)} where its weights were attributed; a regularizer "
                "declares the same blocks at every point."
            )
        dtype = np.asarray(point).dtype
        total = 0.0
        for weight, (_, residual, _) in zip(self.block_weights, declared):
            if weight == 0.0:
                continue
            residual = np.ravel(np.asarray(residual, dtype=dtype))
            total += weight * (0.5 * float(residual @ residual))
        return total

    def penalty_gradient(self, point: np.ndarray) -> np.ndarray:
        """The gradient of ``penalty`` at ``point``: every block's ``J_b^T r_b`` times its weight.

        Each block of ``view.regularization.block_rows`` adds its slope to the
        joint columns it addresses; a regularizer without rows gives zeros.
        """
        point = np.asarray(point)
        gradient = np.zeros(point.shape, dtype=point.dtype)
        regularizer = self.view.regularization
        if int(np.size(regularizer.residual(jax.device_put(point, self.device)))) == 0:
            return gradient
        declared = tuple(regularizer.block_rows(jax.device_put(point, self.device)))
        weights = self.block_weights
        if weights is None:
            weights = (1.0,) * len(declared)
        elif len(declared) != len(weights):
            raise ValueError(
                f"the regularizer's block_rows(x) declares {len(declared)} blocks at this point "
                f"and {len(weights)} where its weights were attributed; a regularizer declares "
                "the same blocks at every point."
            )
        for weight, (columns, residual, jacobian) in zip(weights, declared):
            if weight == 0.0:
                continue
            residual = np.ravel(np.asarray(residual, dtype=gradient.dtype))
            jacobian = np.asarray(jacobian, dtype=gradient.dtype).reshape(residual.size, -1)
            gradient[np.asarray(columns, dtype=np.int64)] += weight * (jacobian.T @ residual)
        return gradient

    def trial_cost(self, results: Any, point: np.ndarray, *, member: int = 0) -> float:
        """The weighted lane costs of a ``value`` evaluation and the regularizer at ``point``.

        ``member`` is the population member of ``results`` whose lane costs
        are summed; a fit's own requests hold the one member 0.
        """
        rows = np.flatnonzero(np.asarray(results.members) == int(member))
        if rows.shape[0] != 1:
            raise ValueError(
                f"member {int(member)} is not among the members of these results "
                f"{np.asarray(results.members).tolist()}; ask for one of them."
            )
        costs = np.asarray(results.cost, dtype=np.float64)[int(rows[0])]
        lanes = float(np.sum(self.omega * costs))
        # The regularizer's cost from its blocks at the trial, which the system there reuses
        # when the trial is accepted: the regularizer is evaluated once per point.
        blocks = self.regularizer_at(point)
        return lanes + (0.0 if blocks is None else float(blocks[1]))

    def owners(self, failures: tuple) -> tuple:
        """The level-1 members of the failed lanes, each once, in lane order."""
        labels = []
        for failure in failures:
            label = self.members[0][int(self.nodes[self.position[failure.lane]][0])]
            if label not in labels:
                labels.append(label)
        return tuple(labels)


# The nested mode's counts kept per member of the cut level.
_MEMBER_COUNTS = (
    "inner_iterations",
    "inner_rejections",
    "inner_unconverged",
    "inner_start_iterations",
    "inner_retreats",
)


def _hold_rungs(rungs: np.ndarray, first_steps: np.ndarray, results: Any, position: dict) -> None:
    """Raise the rung of every (member, lane) that ``results`` solved higher; never lower one.

    ``rungs`` and ``first_steps`` are ``(members, lanes)``, changed in place;
    a row is a population member, ``position`` gives a lane's column. A
    raised rung's base is the first step of the rung-0 solve its rungs
    scaled (its retry record's ``first_step``); a unit that solved at rung 0
    while its lane holds rung 0 keeps the first step its solve took. Results
    that report no rungs leave both as they are.
    """
    for record in getattr(results, "retries", ()):
        member = int(record["member"])
        column = position[str(record["lane"])]
        tried = tuple(int(rung) for rung in record["rungs"])
        if tuple(record["kinds"])[-1] == "ok" and tried[-1] > rungs[member, column]:
            rungs[member, column] = tried[-1]
            first_steps[member, column] = float(record["first_step"])
    status = getattr(results, "status", None)
    first = getattr(getattr(results, "work", None), "first_step", None)
    if status is None or first is None:
        return
    members = np.asarray(results.members, dtype=np.int64)
    columns = np.asarray([position[str(lane)] for lane in results.lane_ids], dtype=np.int64)
    kinds = np.asarray(status.kind)
    produced = np.asarray(status.rung)
    first = np.asarray(first, dtype=np.float64)
    keep = (kinds == 0) & (produced == 0) & (rungs[np.ix_(members, columns)] == 0)
    row, column = np.nonzero(keep)
    first_steps[members[row], columns[column]] = first[row, column]


class _Work:
    """The counts, the failure log and the rungs of one fit, reported as ``OptResult.work``.

    Every lane holds a rung, 0 at the start of a fresh fit: a lane that a
    request solved at a higher rung than it asked for is asked for at that rung
    from then on, with the first step of the rung-0 solve that rung scales as
    its base; a lane that solved at rung 0 keeps the first step its solve took
    as its base. A rung is never lowered.
    """

    COUNTS = (
        "attempts",
        "accepted",
        "rejected",
        "blocks",
        "value",
        "resume_blocks",
        "predicate_rejections",
        "invalid_trials",
        "solves",
    )

    def __init__(self, route: _LaneRoute):
        for name in self.COUNTS:
            setattr(self, name, 0)
        self.route = route
        self.lane_failures = np.zeros((len(route.ids),), dtype=np.int64)
        self.held: tuple = ()
        self.log: list = []
        # The rung each lane is asked for, the first step of its rung-0 solve, the number
        # of evaluations that solved it again, and the consistency re-solves made.
        self.lane_rungs = np.zeros((len(route.ids),), dtype=np.int64)
        self.lane_first_steps = np.zeros((len(route.ids),), dtype=np.float64)
        self.retries = np.zeros((len(route.ids),), dtype=np.int64)
        self.rung_switch_resolves = 0
        # The retry records and the rung switches since the last attempt record.
        self.attempt_retries: list = []
        self.attempt_switches: list = []
        # The nested mode's counts per member of the cut level, and the number of attempts
        # made with the polish switched off; None in the joint mode.
        self.inner_members: Any = None
        self.inner_iterations: Any = None
        self.inner_rejections: Any = None
        self.inner_unconverged: Any = None
        self.inner_start_iterations: Any = None
        self.inner_retreats: Any = None
        self.polish_skipped: Any = None

    def nest(self, members: tuple) -> None:
        """Count the inner iterations of every member of the cut level from here on."""
        self.inner_members = tuple(members)
        for name in _MEMBER_COUNTS:
            setattr(self, name, np.zeros((len(members),), dtype=np.int64))
        self.polish_skipped = 0

    def restore(self, state: Any) -> None:
        """Continue the counts and the failure log of the saved state of the same fit."""
        for name in self.COUNTS:
            setattr(self, name, int(np.asarray(state[f"work.{name}"])))
        saved = dict(
            zip(
                (str(lane) for lane in np.ravel(state["work.lane_ids"])),
                (int(count) for count in np.ravel(state["work.lane_failures"])),
            )
        )
        self.lane_failures = np.asarray(
            [saved.get(lane, 0) for lane in self.route.ids], dtype=np.int64
        )
        attempts = np.ravel(state["work.failure_attempt"])
        # A state saved without the rungs of its failures logs none.
        try:
            tried = [_unpadded(row) for row in np.asarray(state["work.failure_rungs"])]
        except (KeyError, TypeError):
            tried = [()] * attempts.size
        # A state saved without the ignored marks logs every failure as counted.
        try:
            ignored = [bool(value) for value in np.ravel(state["work.failure_ignored"])]
        except (KeyError, TypeError):
            ignored = [False] * attempts.size
        self.log = [
            (
                int(attempt),
                _Failure(
                    str(lane),
                    str(kind),
                    float(reached),
                    tuple(str(node) for node in nodes),
                    rungs,
                    skipped,
                ),
                bool(inner),
            )
            for attempt, lane, kind, reached, nodes, inner, rungs, skipped in zip(
                attempts,
                np.ravel(state["work.failure_lane"]),
                np.ravel(state["work.failure_kind"]),
                np.ravel(state["work.failure_t"]),
                np.asarray(state["work.failure_nodes"]),
                np.ravel(state["work.failure_inner"]),
                tried,
                ignored,
            )
        ]
        # A state saved without the rungs resumes with every lane at rung 0.
        ids = [str(lane) for lane in np.ravel(state["work.lane_ids"])]
        for name, dtype in (
            ("lane_rungs", np.int64),
            ("lane_first_steps", np.float64),
            ("retries", np.int64),
        ):
            try:
                values = dict(zip(ids, np.ravel(np.asarray(state[f"work.{name}"]))))
            except (KeyError, TypeError):
                values = {}
            setattr(
                self,
                name,
                np.asarray([values.get(lane, 0) for lane in self.route.ids], dtype=dtype),
            )
        self.rung_switch_resolves = _saved_count(state, "work.rung_switch_resolves")
        if self.inner_members is not None:
            for name in ("inner_iterations", "inner_rejections", "inner_unconverged"):
                setattr(self, name, np.array(state[f"work.{name}"], dtype=np.int64))
            # A state saved without the start's iterations, the retreats or the skipped
            # polishes resumes with none.
            for name in ("inner_start_iterations", "inner_retreats"):
                try:
                    saved = np.array(state[f"work.{name}"], dtype=np.int64)
                except (KeyError, TypeError):
                    saved = np.zeros((len(self.inner_members),), dtype=np.int64)
                setattr(self, name, saved)
            self.polish_skipped = _saved_count(state, "work.polish_skipped")

    def record(self, attempt: int, failures: tuple, *, inner: bool) -> None:
        """Log failures the run survives; a failure marked ``ignored`` is logged as such."""
        for failure in failures:
            self.lane_failures[self.route.position[failure.lane]] += 1
            self.log.append((attempt, failure, inner))

    def attempted(self, accepted: bool) -> None:
        """Count an attempt, its trial accepted or rejected."""
        if accepted:
            self.accepted += 1
        else:
            self.rejected += 1
        self.attempts += 1

    def request(self, lanes: Any) -> dict:
        """The keywords of a request for ``lanes`` (``None``: every lane) at the rungs held.

        A request whose lanes all hold rung 0 names no rung.
        """
        route = self.route
        keywords: dict = {}
        if lanes is None:
            positions = np.arange(len(route.ids))
        else:
            positions = np.asarray([route.position[lane] for lane in lanes], dtype=np.int64)
            keywords["lanes"] = tuple(lanes)
        rungs = self.lane_rungs[positions]
        if np.any(rungs > 0):
            keywords["rungs"] = rungs[None, :].copy()
            keywords["first_steps"] = self.lane_first_steps[positions][None, :].copy()
        return keywords

    def hold(self, results: Any) -> None:
        """Hold the rungs ``results`` raised (``_hold_rungs``, the fit's one member 0).

        Every retry record counts for its lane and joins the records of the
        attempt.
        """
        for record in getattr(results, "retries", ()):
            self.retries[self.route.position[str(record["lane"])]] += 1
            self.attempt_retries.append(record)
        _hold_rungs(
            self.lane_rungs[None, :], self.lane_first_steps[None, :], results, self.route.position
        )

    def attempt_records(self) -> dict:
        """The retry records and rung switches since the last attempt record, then none."""
        records = {
            "retries": tuple(self.attempt_retries),
            "switches": tuple(self.attempt_switches),
        }
        self.attempt_retries, self.attempt_switches = [], []
        return records

    def mapping(self) -> MappingProxyType:
        """The work so far as a read-only mapping of host arrays."""
        route = self.route
        entries = {name: np.asarray(getattr(self, name), dtype=np.int64) for name in self.COUNTS}
        entries["lane_ids"] = _strings(route.ids)
        entries["lane_failures"] = self.lane_failures.copy()
        entries["held"] = _string_rows(
            [(route.names[level - 1], route.members[level - 1][node]) for level, node in self.held],
            2,
        )
        entries["failure_attempt"] = np.asarray([entry[0] for entry in self.log], dtype=np.int64)
        entries["failure_lane"] = _strings(entry[1].lane for entry in self.log)
        entries["failure_kind"] = _strings(entry[1].kind for entry in self.log)
        entries["failure_t"] = np.asarray(
            [entry[1].t_reached for entry in self.log], dtype=np.float64
        )
        entries["failure_nodes"] = _string_rows([entry[1].nodes for entry in self.log], route.depth)
        entries["failure_inner"] = np.asarray([entry[2] for entry in self.log], dtype=bool)
        entries["failure_ignored"] = np.asarray(
            [entry[1].ignored for entry in self.log], dtype=bool
        )
        entries["failure_rungs"] = _padded_rows(
            [entry[1].rungs for entry in self.log], -1, np.int64
        )
        entries["lane_rungs"] = self.lane_rungs.copy()
        entries["lane_first_steps"] = self.lane_first_steps.copy()
        entries["retries"] = self.retries.copy()
        entries["rung_switch_resolves"] = np.asarray(self.rung_switch_resolves, dtype=np.int64)
        if self.inner_members is not None:
            entries["inner_members"] = _strings(self.inner_members)
            for name in _MEMBER_COUNTS:
                entries[name] = getattr(self, name).copy()
            entries["polish_skipped"] = np.asarray(self.polish_skipped, dtype=np.int64)
        return MappingProxyType(entries)


def _record(
    rho: float,
    lam: float,
    accepted: bool,
    trial: np.ndarray,
    failures: tuple,
    depth: int,
    *,
    retries: tuple = (),
    switches: tuple = (),
):
    """The attempt just made, as the ``record.*`` entries of the state.

    ``retries`` are the engine's records of the lanes its requests solved
    again; ``switches`` the consistency re-solves, one row per lane: the lane,
    its old and new rung, and its cost at the current point before and after.
    """
    return {
        "rho": np.asarray(rho, dtype=np.float64),
        "lam": np.asarray(lam, dtype=np.float64),
        "accepted": np.asarray(bool(accepted)),
        "trial": np.array(trial, dtype=np.float64),
        "failed_lanes": _strings(failure.lane for failure in failures),
        "failed_kinds": _strings(failure.kind for failure in failures),
        "failed_nodes": _string_rows([failure.nodes for failure in failures], depth),
        "rungs_tried": _padded_rows([failure.rungs for failure in failures], -1, np.int64),
        "rung_switches": _string_rows(
            [
                (lane, old, new, repr(float(before)), repr(float(after)))
                for lane, old, new, before, after in switches
            ],
            5,
        ),
        "retry_lanes": _strings(record["lane"] for record in retries),
        "retry_rungs": _padded_rows([record["rungs"] for record in retries], -1, np.int64),
        "retry_kinds": _padded_rows([record["kinds"] for record in retries], "", str),
        "retry_first_steps": np.asarray(
            [record["first_step"] for record in retries], dtype=np.float64
        ),
        "retry_seconds": _padded_rows(
            [record["seconds"] for record in retries], np.nan, np.float64
        ),
    }


def _stop_report(
    status: Any, cost_changes: list, decreases: list, stalled: Any = None, budget: Any = None
) -> tuple:
    """The message and the ``info`` entries of a fit, from its window of rejected trials.

    A fit that ends at ``max_steps`` or stalled reports how many rejected
    trials its window holds and, when there are any, the medians of their cost
    changes and predicted decreases, each relative to the cost scale
    ``max(|cost|, cost_atol / cost_tol)`` at its attempt. The
    ``max_steps`` message says which ``cost_tol`` would end the fit at that
    level; ``budget``, for a fit that ``max_evaluations`` ended, states the
    evaluations it made right after the status. A stalled fit's message is
    ``stalled`` (what ended the fit and its numbers) followed by the medians.
    Every other fit reports nothing and its message is the status value.
    """
    if status not in (TerminationStatus.MAX_STEPS, TerminationStatus.STALLED):
        return status.value, {}
    head = status.value if budget is None else f"{status.value}: {budget}"
    lead = stalled if status is TerminationStatus.STALLED else head
    k = len(cost_changes)
    entries: dict = {"last_rejected_trials": int(k)}
    if k == 0:
        return lead, entries
    a = float(np.median(np.asarray(cost_changes, dtype=np.float64)))
    p = float(np.median(np.asarray(decreases, dtype=np.float64)))
    if np.isfinite(a):
        entries["median_rejected_cost_change"] = a
    if np.isfinite(p):
        entries["median_rejected_predicted_decrease"] = p
    if not (np.isfinite(a) and np.isfinite(p)):
        return lead, entries
    medians = (
        f"over the last {k} rejected trial(s) the median size of the cost change was {a:.2g} "
        f"and the median predicted decrease was {p:.2g}, both relative to the cost scale "
        "max(|cost|, cost_atol / cost_tol)"
    )
    if status is TerminationStatus.STALLED:
        return f"{lead}; {medians}", entries
    t = max(a, p)
    prefix = f"{status.value}:" if budget is None else f"{head};"
    message = (
        f"{prefix} {medians}; when the cost is evaluated with noise of this size, "
        f"cost_tol={t:.2g} or larger ends the fit at that noise level."
    )
    return message, entries


# The status of a loop-ending event that the verdict does not decide.
_STOP_STATUS = {
    "gradient_tolerance": TerminationStatus.GRAD_TOL,
    "stationarity": TerminationStatus.GRAD_TOL,
    "derivative_accuracy": TerminationStatus.DERIVATIVE_ACCURACY,
    "max_steps": TerminationStatus.MAX_STEPS,
    "max_evaluations": TerminationStatus.MAX_STEPS,
    "accepted_steps": TerminationStatus.ACCEPTED_STEPS,
    "callback": TerminationStatus.STOPPED,
    "invalid_trials": TerminationStatus.STALLED,
}

# What a step or a cost test found within its tolerance, in a stalled fit's message.
_ENDED_BY = {"step_tolerance": "the step", "cost_tolerance": "the cost change"}


def _stationarity_limit(tolerance: Any, cost_tol: float, noise: float) -> float:
    """``stationarity_tol``, or for ``None`` the yardstick ``max(cost_tol, 3 * noise)``."""
    return max(float(cost_tol), 3.0 * float(noise)) if tolerance is None else float(tolerance)


def _verdict(
    stop: str,
    *,
    sigma: float,
    yardstick: float,
    lock: int,
    members: bool = True,
    floor_sigma: float = 0.0,
    floor_known: bool = False,
    jump_yardstick: Any = None,
    jump_members: bool = True,
) -> tuple:
    """The status and the stall reason of a loop that the event ``stop`` ended.

    A step test, a cost test or a run of noise-dominated trials ends the fit
    converged only at a point where the model predicts no more than the
    evaluation can show (``sigma <= yardstick``, and ``members``: in the
    nested mode every member's model too, against the members' yardstick) and
    no invalid trial shrank the radius (the growth lock is 0). Failing that,
    when the gradient floor is known (``floor_known``: a small rejected trial
    measured it), the growth lock is 0 and the model predicts no more than
    the gradient's error along its full step could take back
    (``sigma <= floor_sigma``, and ``members``, every member within its own
    floor), the fit converged to the accuracy of its derivatives. Failing
    that, a step test or a cost test ends the fit converged, with its own
    status, when every rejected trial in the window met a jump of the
    computed cost (``jump_yardstick``, ``max(cost_tol, 3 * the median
    jump)``, is a number), the growth lock is 0 and the model predicts no
    more than that yardstick (and ``jump_members``, every member within it
    too): the decrease left is below what any trial can show. Otherwise
    the fit is stalled, for invalid trials when the lock is set, else for the
    evaluation noise when the noise-dominated trials ended it, else because
    the radius collapsed. A run of invalid trials ends the fit stalled for
    invalid trials; every other event gives its own status.
    """
    if stop not in ("step_tolerance", "cost_tolerance", "evaluation_noise"):
        status = _STOP_STATUS[stop]
        return status, ("invalid_trials" if status is TerminationStatus.STALLED else None)
    if lock == 0 and sigma <= yardstick and members:
        if stop == "step_tolerance":
            return TerminationStatus.STEP_TOL, None
        return TerminationStatus.COST_TOL, None
    if lock == 0 and floor_known and sigma <= floor_sigma and members:
        return TerminationStatus.DERIVATIVE_ACCURACY, None
    if (
        stop != "evaluation_noise"
        and lock == 0
        and jump_yardstick is not None
        and sigma <= jump_yardstick
        and jump_members
    ):
        if stop == "step_tolerance":
            return TerminationStatus.STEP_TOL, None
        return TerminationStatus.COST_TOL, None
    if lock:
        return TerminationStatus.STALLED, "invalid_trials"
    if stop == "evaluation_noise":
        return TerminationStatus.STALLED, "evaluation_noise"
    return TerminationStatus.STALLED, "radius_collapsed"


def _last_invalid(log: list) -> tuple:
    """The failed lanes of the fit's last invalid trial, from its failure log; empty when none.

    An ignored failure (a lane of weight zero) made no trial invalid and is skipped.
    """
    rows: list = []
    last = None
    for attempt, failure, inner in reversed(log):
        if inner or failure.ignored:
            continue
        if last is None:
            last = attempt
        if attempt != last:
            break
        rows.append(failure)
    return tuple(reversed(rows))


def _stalled_message(
    reason: str,
    stop: str,
    *,
    consecutive: int,
    failures: tuple,
    radius: float,
    sigma: float,
    noise: float,
    ceiling: Any,
    member: Any = None,
) -> str:
    """What ended a stalled fit and its numbers, to two significant digits.

    ``member``, in the nested mode, is ``(level, label, number, yardstick)`` of
    the member whose model predicts the most beyond the members' yardstick, or
    ``None``; its clause ends the message.
    """
    clause = ""
    if member is not None:
        level, label, number, limit = member
        clause = (
            f"; member {label!r} of level {level!r} predicts a further relative decrease of "
            f"{number:.2g} of its own cost, above {limit:.2g}"
        )
    return (
        _stalled_text(reason, stop, consecutive, failures, radius, sigma, noise, ceiling) + clause
    )


def _stalled_text(
    reason: str,
    stop: str,
    consecutive: int,
    failures: tuple,
    radius: float,
    sigma: float,
    noise: float,
    ceiling: Any,
) -> str:
    """``_stalled_message`` without the member's clause."""
    if reason == "evaluation_noise":
        tolerance = (
            "" if ceiling is None else f" (lane solvers at a relative tolerance of {ceiling:.2g})"
        )
        return (
            f"stalled: the model predicts a further relative decrease of {sigma:.2g} while the "
            f"cost's evaluation jitter was measured at about {noise:.2g}{tolerance}; a tighter "
            "lane tolerance resolves the optimum further"
        )
    if reason == "radius_collapsed":
        return (
            f"stalled: {_ENDED_BY[stop]} fell within its tolerance while the model still "
            f"predicts a relative decrease of {sigma:.2g}; the radius is {radius:.2g}"
        )
    where = "one" if stop == "invalid_trials" else "invalid trial"
    failed = (
        f"; at the last {where} the lane solve failed for experiment(s) "
        + _listed(
            f"{failure.lane} ({failure.kind}) at t = {failure.t_reached:.2g}"
            f"{_rungs_text(failure.rungs)}"
            for failure in failures
        )
        if failures
        else ""
    )
    if stop == "invalid_trials":
        event = f"the last {consecutive} trial(s) in a row were invalid"
    elif stop == "evaluation_noise":
        event = (
            "three rejected trials in a row were within the noise of the cost's evaluation "
            "after an invalid trial shrank the radius"
        )
    else:
        event = (
            f"{_ENDED_BY[stop]} fell within its tolerance after an invalid trial shrank the radius"
        )
    return (
        f"stalled: {event}{failed}; the radius is {radius:.2g}; the model predicts a further "
        f"relative decrease of {sigma:.2g}"
    )


def _full_step_decrease(
    system: Any, gradient: np.ndarray, scale: Any, zero: tuple, addition: Any = None
) -> float:
    """The decrease the model predicts for its full step.

    ``zero`` is the solve at zero damping in the coordinates ``z = scale * s``
    (``solve_scaled(system, scale, 0.0, addition)``), and the full step is
    ``s = z / scale``: zero on the fixed coordinates, under the system's
    equality rows. The model is the one that solve minimizes, its scaled matrix
    carrying ``addition`` on its diagonal. The decrease is ``nan`` when that
    solve is not finite (the matrix is singular at this point).
    """
    if isinstance(zero, elimination.AttemptSolve) and zero.answers(scale, zero.radius, addition):
        # The attempt's program predicted it on the device, from the same solve.
        return zero.decrease
    z, q_squared = np.asarray(zero[0]), float(zero[1])
    if not (np.all(np.isfinite(z)) and np.isfinite(q_squared)):
        return float("nan")
    step = (1.0 / np.asarray(scale, dtype=z.dtype)) * z
    quadratic = elimination.level_quadratic(system, step)
    if addition is not None:
        quadratic += float(np.asarray(addition, dtype=np.float64) @ (z * z))
    return -(float(gradient @ step) + 0.5 * quadratic)


def _stationarity(
    system: Any,
    gradient: np.ndarray,
    scale: Any,
    zero: tuple,
    relative_to: float,
    addition: Any = None,
) -> float:
    """The decrease the model predicts for its full step, relative to the cost scale.

    The decrease is ``_full_step_decrease``'s, divided by ``relative_to``, the
    cost scale ``max(|cost|, cost_atol / cost_tol)`` at the point
    (``cost_scale``); the number is ``nan`` when the solve at zero damping is
    not finite.
    """
    return _full_step_decrease(system, gradient, scale, zero, addition) / relative_to


class _FullStep(NamedTuple):
    """The model's full step at a point, as the gradient floor reads it.

    ``norm`` is its norm in the subproblem's coordinates ``z = scale * s``,
    the norm the radius is measured in; ``length`` its norm in parameter
    units; ``decrease`` the decrease the model predicts for it, in cost units
    (``nan`` when the solve at zero damping is not finite); ``scale`` the
    coordinates' scale.
    """

    norm: float
    length: float
    decrease: float
    scale: Any


def _full_step(
    system: Any, gradient: np.ndarray, scale: Any, zero: tuple, addition: Any = None
) -> _FullStep:
    """The full step of ``zero``, the solve at zero damping, with ``_full_step_decrease``'s decrease."""
    z = np.asarray(zero[0])
    with np.errstate(all="ignore"):
        length = float(np.linalg.norm(z / np.asarray(scale, dtype=z.dtype)))
    decrease = _full_step_decrease(system, gradient, scale, zero, addition)
    return _FullStep(float(np.linalg.norm(z)), length, decrease, scale)


def _subproblem(x: np.ndarray, gradient: np.ndarray, scale: np.ndarray, bounds: Any) -> Any:
    """The coordinates of the trust-region subproblem at ``x``.

    Without bounds they are ``z = scale * s`` and the matrix has no diagonal
    addition (``addition`` is ``None``); with bounds they are the Coleman-Li
    coordinates of ``_bounds.coleman_li_scaling``.
    """
    if bounds is None:
        ones = np.ones_like(x)
        return box.Scaling(v=ones, dv=np.zeros_like(x), scale=scale, addition=None, theta=1.0)
    return box.coleman_li_scaling(x, gradient, bounds[0], bounds[1], scale)


def _bounded_gradient(gradient: np.ndarray, coordinates: Any, bounds: Any) -> np.ndarray:
    """The gradient the gradient test reads: ``g * v`` under bounds, ``g`` without."""
    return gradient if bounds is None else gradient * coordinates.v


def _choose(
    system: Any,
    x: np.ndarray,
    gradient: np.ndarray,
    coordinates: Any,
    step: np.ndarray,
    radius: float,
    bounds: tuple,
    *,
    equality: bool,
) -> tuple:
    """The trial step under the box, from the subproblem's solution ``step``.

    ``_bounds.select_step`` on the scaled model of ``system`` with its diagonal
    addition; with equality rows only its first candidate, the step shortened
    to stay inside, which keeps the rows. Returns ``(step, scaled_step,
    predicted)``.
    """
    scale, addition = coordinates.scale, coordinates.addition
    with np.errstate(all="ignore"):
        d = 1.0 / scale
        scaled_step = scale * step

    def quadratic_form(vector: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            return elimination.level_quadratic(system, d * vector) + float(
                addition @ (vector * vector)
            )

    g_h = d * gradient
    lower, upper = bounds
    if equality:
        return box.shortened_step(
            x, quadratic_form, g_h, step, scaled_step, lower, upper, coordinates.theta
        )
    reach = max(float(radius), float(np.linalg.norm(scaled_step)))
    return box.select_step(
        x, quadratic_form, g_h, step, scaled_step, d, reach, lower, upper, coordinates.theta
    )


def _trial_point(
    system: Any,
    x: np.ndarray,
    gradient: np.ndarray,
    coordinates: Any,
    radius: float,
    *,
    first: tuple,
    fixed: np.ndarray,
    bounds: Any,
    equality: bool,
) -> tuple:
    """The step of radius ``radius`` from ``x`` in the subproblem's ``coordinates``, and its trial.

    The step is ``level_secular_step``'s on ``system`` from its solve at zero
    damping ``first``, with the coordinates ``fixed`` held. Without bounds the
    trial is ``x + step``; with them it is ``x`` plus the best of the step,
    the step reflected at the sides it crosses and the Cauchy step, by the
    model's value (``_choose``; ``equality``: the system has equality rows),
    kept strictly inside the box. Returns ``(trial, scaled_norm, predicted,
    multiplier, solves)``: ``scaled_norm`` the chosen step's norm in the
    subproblem's coordinates, ``predicted`` the decrease the model predicts
    for it.
    """
    if (
        isinstance(first, elimination.AttemptSolve)
        and not first.failed
        and first.answers(coordinates.scale, radius, coordinates.addition)
    ):
        # The attempt's program solved the step of this radius with its solve at zero.
        proposed, predicted = first.step.copy(), first.predicted
        multiplier, solves = first.multiplier, first.solves
    else:
        # The search on the host, from the solve at zero damping; a search the program
        # could not bracket is refused here, where the step is taken.
        proposed, predicted, multiplier, solves = elimination.level_secular_step(
            system, coordinates.scale, radius, first=first, addition=coordinates.addition
        )
    if fixed.size:
        proposed[fixed] = 0.0
    if bounds is None:
        trial = x + proposed
        scaled_norm = float(np.linalg.norm(coordinates.scale * proposed))
    else:
        chosen, chosen_scaled, predicted = _choose(
            system, x, gradient, coordinates, proposed, radius, bounds, equality=equality
        )
        trial = box.make_strictly_feasible(x + chosen, bounds[0], bounds[1], rstep=0)
        scaled_norm = float(np.linalg.norm(chosen_scaled))
    return trial, scaled_norm, predicted, multiplier, solves


def _trial_valid(view: Any, point: np.ndarray) -> bool:
    """The trial predicate of ``view`` at ``point``; a predicate that raises rejects it."""
    try:
        return trial_is_valid(view, point)
    except Exception:  # noqa: BLE001 -- a predicate that raises rejects the trial
        return False


def _noise_ceiling(route: Any) -> Any:
    """The route's ceiling of the evaluation noise as a float, or ``None`` when it is unknown."""
    ceiling = getattr(route, "evaluation_noise", None)
    return None if ceiling is None else float(ceiling)


def _noise_trial_limit(solver: Any, route: Any) -> Any:
    """The largest relative predicted decrease of a trial counted toward the evaluation noise.

    It is ``noise_trial_limit`` times the route's ceiling of the evaluation
    noise, or ``None`` (no bound) when either is unset.
    """
    ceiling = _noise_ceiling(route)
    if solver.noise_trial_limit is None or ceiling is None:
        return None
    return float(solver.noise_trial_limit) * ceiling


# How far the a priori bound of the gradient's error exceeds the corrector's band times the
# size of the lanes' gradient terms. Calibrated on a three-particle test model as twice the
# largest measured ratio of the gradient's error (the centered difference of the computed
# values along the model's step, the gradient and a random direction, minus the gradient's
# slope) to the band times the terms: 0.24 at rtol 3e-4 and 0.51 at rtol 3e-5, where the
# error falls more slowly than the tolerance because the step grid's own parameter dependence
# does not follow the corrector's band; the margin is taken at the tighter tolerance.
_FLOOR_FACTOR = 1.0

# The band within which a lane's nonlinear corrector accepts its iterate, per unit of the
# solver's rtol and newton_tol_factor, relative: the derivative at that iterate differs from
# the slope of the computed values by about this band.
_BAND = 0.33


def _lane_tolerances(route: Any) -> tuple:
    """Per lane, its experiment solver's ``rtol`` and ``newton_tol_factor``, as host arrays.

    Both are read as ``LoweredStudy.evaluation_noise`` reads ``rtol``: from
    the solver of the lane's experiment, the largest component of a vector.
    A solver without ``rtol`` gives ``nan``, one without ``newton_tol_factor``
    gives 1.
    """
    problem = route.problem
    count = len(route.ids)
    rtols = np.full((count,), np.nan)
    factors = np.ones((count,))
    experiments = tuple(getattr(getattr(problem, "study", None), "experiments", None) or ())
    if not experiments:
        return rtols, factors
    layout = lane_layout(problem.diagnostic_problem)
    positions = _lane_positions(problem.plan(), route.ids)
    for lane, index in enumerate(np.asarray(layout.experiments, dtype=np.int64)[positions]):
        solver = getattr(experiments[int(index)], "solver", None)
        rtol = getattr(solver, "rtol", None)
        if rtol is not None:
            rtols[lane] = float(np.max(np.asarray(rtol, dtype=np.float64)))
        factor = getattr(solver, "newton_tol_factor", None)
        if factor is not None:
            factors[lane] = float(np.max(np.asarray(factor, dtype=np.float64)))
    return rtols, factors


def _lane_bands(route: Any) -> Any:
    """Every lane's corrector band ``0.33 * newton_tol_factor * rtol``, or ``None``.

    A lane whose solver has no ``rtol`` has the band 0 and adds nothing to the
    bound of the gradient's error; ``None`` when no lane's solver has one,
    and the fit then has no gradient floor.
    """
    rtols, factors = _lane_tolerances(route)
    known = np.isfinite(rtols)
    if not known.any():
        return None
    return np.where(known, _BAND * factors * np.where(known, rtols, 0.0), 0.0)


def _lane_columns(route: Any) -> np.ndarray:
    """Per lane, the joint column of each of its narrow coordinates, shape ``(lanes, width)``.

    A lane's narrow coordinates are the shared block, then the block of its
    node at every level, as the level reducer reads them; the joint vector
    holds the shared block, then every node of level 1 in node order, and so
    on down the levels.
    """
    widths = tuple(int(width) for width in route.tables.widths)
    counts = tuple(int(count) for count in route.tables.counts)
    nodes = np.asarray(route.nodes, dtype=np.int64)
    lanes = nodes.shape[0]
    parts = [np.broadcast_to(np.arange(widths[0], dtype=np.int64), (lanes, widths[0]))]
    offset = widths[0]
    for level in range(1, len(widths)):
        width = widths[level]
        parts.append(
            offset + nodes[:, level - 1 : level] * width + np.arange(width, dtype=np.int64)
        )
        offset += counts[level - 1] * width
    return np.concatenate(parts, axis=1)


class _GradientFloor:
    """The a priori bound of the gradient's error at a fit's current point.

    The bound is ``sum_l a_l * t_l * sqrt(2 * c_l)`` over the lanes, in cost
    per unit of scaled step length: ``c_l`` is the lane's cost and
    ``t_l = sqrt(sum_i N_l[i, i] / scale_i**2)`` over its free coordinates,
    ``N_l`` its block, so that ``t_l * sqrt(2 * c_l)`` bounds the size of the
    lane's gradient terms ``|J_l|^T |r_l|`` in the scaled coordinates;
    ``a_l`` is ``kappa * omega_l * band_l`` under ``gradient_floor="auto"``
    (``omega_l`` the lane's weight, ``band_l`` its corrector's band) and
    ``value * omega_l`` under a number. The regularizer's rows are exact and
    add nothing. ``current`` holds the lanes' diagonals and costs of the
    blocks request at the current point and ``latest`` those of the latest
    blocks request; the bound is ``None`` without them, without a floor, or
    when the results carry neither the lanes' blocks nor their diagonals.
    """

    def __init__(self, solver: Any, route: Any):
        """Read the solver's ``gradient_floor`` and the lanes' weights, bands and columns."""
        setting = solver.gradient_floor
        self.route = route
        self.given = None if setting is None or isinstance(setting, str) else float(setting)
        self.coefficients = None
        if setting is not None:
            omega = np.asarray(route.omega, dtype=np.float64)
            if self.given is not None:
                self.coefficients = self.given * omega
            else:
                bands = _lane_bands(route)
                if bands is not None:
                    self.coefficients = _FLOOR_FACTOR * omega * bands
        self.columns = None if self.coefficients is None else _lane_columns(route)
        # A fixed coordinate never moves: its share of a gradient's error is never taken.
        self.free = np.ones((route.n,), dtype=np.float64)
        self.free[np.asarray(route.fixed, dtype=np.int64)] = 0.0
        self.current: Any = None
        self.latest: Any = None

    def record(self, results: Any) -> Any:
        """Keep the lanes' terms of a ``blocks`` (or ``hessian``) request as ``latest``; return them.

        The terms are host arrays ``(lanes, width)`` and ``(lanes,)``: every
        lane's diagonal of its matrix and its cost, of member 0. The diagonals
        come from the lanes' blocks in one process, and from the diagonals the
        processes gather beside the level sums (``Results.normal_diagonal``)
        when the lanes run in several. Under ``hessian="true"`` the diagonal of
        the exact Hessian stands in for that of the Gauss-Newton matrix, its
        negative entries counted as 0. ``None`` without a floor or when the
        results carry neither the lanes' blocks nor their diagonals.
        """
        self.latest = None
        if self.columns is None:
            return None
        rows = np.flatnonzero(np.asarray(results.members) == 0)
        if rows.shape[0] != 1:
            return None
        row = int(rows[0])
        normal = getattr(results, "normal", None)
        if normal is not None:
            matrices = np.asarray(normal)[row]
            if matrices.ndim != 3 or matrices.shape[-2] != matrices.shape[-1]:
                return None
            diagonals = np.diagonal(matrices, axis1=-2, axis2=-1)
        else:
            gathered = getattr(results, "normal_diagonal", None)
            if gathered is None:
                return None
            diagonals = np.asarray(gathered)[row]
        if diagonals.shape != self.columns.shape:
            return None
        diagonals = np.array(diagonals, dtype=np.float64)
        costs = np.array(np.asarray(results.cost, dtype=np.float64)[row])
        self.latest = (diagonals, costs)
        return self.latest

    def bound(self, scale: Any) -> Any:
        """The bound at the current point in the coordinates ``z = scale * s``, or ``None``."""
        if self.coefficients is None or self.current is None:
            return None
        diagonals, costs = self.current
        with np.errstate(all="ignore"):
            inverse = self.free / np.square(np.asarray(scale, dtype=np.float64))
            inverse = np.where(np.isfinite(inverse), inverse, 0.0)
            terms = np.maximum(diagonals, 0.0) * inverse[self.columns]
            lengths = np.sqrt(np.sum(terms, axis=1))
            value = float(
                np.sum(self.coefficients * lengths * np.sqrt(2.0 * np.maximum(costs, 0.0)))
            )
        return value if np.isfinite(value) else None

    def message(self, sigma: float, floor_sigma: float) -> str:
        """The message of a fit that converged to the accuracy of its derivatives."""
        if self.given is not None:
            source = f"from the given derivative accuracy {self.given:.2g}"
        else:
            rtols, _ = _lane_tolerances(self.route)
            known = rtols[np.isfinite(rtols)]
            spread = "up to " if known.size and np.any(known != known[0]) else ""
            top = float(np.max(known)) if known.size else float("nan")
            source = f"from the solver tolerance rtol {spread}{top:.2g}"
        return (
            "converged to the accuracy of the solver's derivatives: the model's predicted "
            f"decrease ({sigma:.2g}) is below the gradient's accuracy floor ({floor_sigma:.2g}, "
            f"{source}), both relative to the cost scale max(|cost|, cost_atol / cost_tol); a "
            "tighter solver tolerance lowers the "
            "floor in proportion"
        )


class _FloorEnd(NamedTuple):
    """The gradient floor at a fit's end point: what its verdict reads, and its ``info`` entries.

    ``level`` is the floor (``None`` before the first floor sample or without
    a floor), ``sigma`` its cost equivalent along the model's full step,
    relative to the cost scale, and ``members`` every member's floor in
    the nested mode (``None`` otherwise, or while ``level`` is ``None``).
    """

    level: Any
    sigma: float
    members: Any
    entries: dict


def _radius_floor(level: Any, full_norm: float, gradient: Any, scale: Any) -> float:
    """``_floor_radius`` with the gradient's norm in the coordinates ``z = scale * s``.

    0 before the first floor sample, and where the radius is not finite.
    """
    if level is None:
        return 0.0
    with np.errstate(all="ignore"):
        norm = float(np.linalg.norm(np.asarray(gradient) / np.asarray(scale)))
    value = float(_floor_radius(level, full_norm, norm))
    return value if np.isfinite(value) else 0.0


def _refuse_state(state: Any, *, solver: str, mode: str, nested_at: Any, n: int) -> None:
    """Refuse a saved state this solver cannot continue, naming both settings."""
    saved_nested = _saved_value(state, "nested_at")
    raise ValueError(
        f"the saved state is of {_saved_value(state, 'solver')} in {_saved_value(state, 'mode')} "
        f"mode (nested_at={saved_nested or None!r}, {_saved_value(state, 'n')} coordinates) and "
        f"this solver is {solver} in {mode} mode (nested_at={nested_at!r}, {n}); resume with the "
        "same settings, or pass the point only: resume=dataclasses.replace(result, state=None)."
    )


def _check_state(state: Any, solver: TrustRegion, route: _LaneRoute) -> None:
    """Refuse a saved state of another solver, mode, cut level, size or problem."""
    mode = "joint" if solver.nested_at is None else "nested"
    expected = {
        "solver": _LABEL,
        "mode": mode,
        "nested_at": "" if solver.nested_at is None else str(solver.nested_at),
        "n": route.n,
        "fingerprint": route.fingerprint,
    }
    if any(_saved_value(state, key) != value for key, value in expected.items()):
        _refuse_state(state, solver=_LABEL, mode=mode, nested_at=solver.nested_at, n=route.n)


def _scale_candidate(system: Any, strategy: Any, diagonal: np.ndarray) -> np.ndarray:
    """The scale this attempt proposes, from the diagonal of the system's matrix."""
    dtype = diagonal.dtype
    if isinstance(strategy, ConstantScale):
        empty = jnp.zeros((0, diagonal.shape[0]), dtype=dtype)
        return np.sqrt(np.asarray(strategy.damping_diagonal(empty, system.point), dtype=dtype))
    with np.errstate(invalid="ignore"):
        if isinstance(strategy, HessianScale):
            damping = np.maximum(np.abs(diagonal), np.finfo(dtype).eps)
        else:
            damping = np.maximum(diagonal, 1e-10)
    return np.sqrt(damping)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------


class TrustRegion(OptSolver):
    """The trust-region Gauss-Newton method for a hierarchical study, with each step solved cluster by cluster.

    It needs a study lowered with `ds.lower(study)`. Box bounds are kept. Pass `checkpoint=` to
    `ds.inference.optsolve` to save the fit and resume it later.

    Args:
        hessian: `"gauss-newton"` steps on the Gauss-Newton curvature, `"true"` on the exact Hessian.
        nested_at: A level at which the fit splits into an outer step and a refit of each member's own parameters.
        inner: A `TrustRegion` without `nested_at` whose budget and tolerances the member refits use, or `None` for two iterations per member.
        radius_init: The initial trust radius, in scaled parameter units.
        radius_max: The largest trust radius.
        eta: The ratio of achieved to predicted decrease a step must reach to be accepted.
        grad_tol: The gradient-norm tolerance, or `None` to test no gradient.
        step_tol: The step-norm tolerance relative to the parameter norm, or `None` for the dtype's default.
        cost_tol: The tolerance on the change in the objective relative to the objective, or `None` for the dtype's default.
        cost_atol: An absolute floor under the cost-change test, zero when `None`.
        stationarity_tol: The relative predicted decrease below which the fit ends, or `None` for the larger of `cost_tol` and three times the measured noise.
        max_consecutive_invalid_trials: The number of trials in a row with failed solves that ends the fit as `STALLED`.
        noise_trial_limit: A rejected trial counts as noise only when its predicted decrease is within this multiple of the study's evaluation noise.
        noise_relaxed_ratio: If `True`, the ratio of achieved to predicted decrease allows for the measured evaluation noise.
        gradient_floor: The relative accuracy of the derivatives, below which the fit ends `DERIVATIVE_ACCURACY`, `"auto"` to derive it from the solvers' `rtol`, or `None`.
        max_steps: The step budget, counted across resumes.
        max_evaluations: The budget of objective evaluations, which ends the fit as `MAX_STEPS`.
        inner_start_iterations: The number of member refit iterations at the start point, before the first outer step.
        inner_fraction: A member refit ends when its predicted decrease falls to this fraction of the outer step's.
        inner_retreats: How many times a member whose solve fails halves its step before the attempt fails.
        target_accepted_steps: Ends the fit as `ACCEPTED_STEPS` after this many accepted steps.
        x_scale: The parameter scaling, `HessianScale()` when `None`, `JacScale()` or a `ConstantScale`, with `JacScale` refused under `hessian="true"`.
        weights: One weight per member of the first level, or a `(level, weights)` pair, multiplying each member's share of the objective.
        weight_priors: If `True`, the weights also multiply each member's prior and regularization terms.
        constraints: Linear equality constraints on the joint parameter vector, as `LinearConstraint` objects.
        gauges: [Gauge][diff_soup.inference.Gauge] constraints that pin directions the data leave free.
        fixed_indices: The indices of the joint parameter vector held at their starting values.
        callbacks: Callbacks run together with those passed to `ds.inference.optsolve`.
    """

    requires: ClassVar[frozenset] = frozenset({"batched_lanes", RESUME})

    hessian: str = eqx.field(static=True)
    nested_at: Any = eqx.field(static=True)
    inner: Any
    radius_init: Any = eqx.field(static=True)
    radius_max: Any = eqx.field(static=True)
    eta: Any = eqx.field(static=True)
    grad_tol: Any = eqx.field(static=True)
    step_tol: Any = eqx.field(static=True)
    cost_tol: Any = eqx.field(static=True)
    cost_atol: Any = eqx.field(static=True)
    stationarity_tol: Any = eqx.field(static=True)
    max_consecutive_invalid_trials: Any = eqx.field(static=True)
    noise_trial_limit: Any = eqx.field(static=True)
    noise_relaxed_ratio: Any = eqx.field(static=True)
    gradient_floor: Any = eqx.field(static=True)
    max_steps: Any = eqx.field(static=True)
    max_evaluations: Any = eqx.field(static=True)
    inner_start_iterations: Any = eqx.field(static=True)
    inner_fraction: Any = eqx.field(static=True)
    inner_retreats: Any = eqx.field(static=True)
    target_accepted_steps: Any = eqx.field(static=True)
    x_scale: Any
    weights: Any
    weight_priors: bool = eqx.field(static=True)
    constraints: tuple
    gauges: tuple
    fixed_indices: tuple = eqx.field(static=True)
    callbacks: tuple = eqx.field(static=True)

    def __init__(
        self,
        *,
        hessian: Any = "gauss-newton",
        nested_at: Any = None,
        inner: Any = None,
        radius_init: Any = 1.0,
        radius_max: Any = 1e3,
        eta: Any = 1e-4,
        grad_tol: Any = None,
        step_tol: Any = None,
        cost_tol: Any = None,
        cost_atol: Any = None,
        stationarity_tol: Any = None,
        max_consecutive_invalid_trials: Any = 5,
        noise_trial_limit: Any = None,
        noise_relaxed_ratio: Any = False,
        gradient_floor: Any = "auto",
        max_steps: Any = 30,
        max_evaluations: Any = None,
        inner_start_iterations: Any = 0,
        inner_fraction: Any = 1e-2,
        inner_retreats: Any = 3,
        target_accepted_steps: Any = None,
        x_scale: Any = None,
        weights: Any = None,
        weight_priors: Any = True,
        constraints: Any = (),
        gauges: Any = (),
        fixed_indices: Any = (),
        callbacks: Any = (),
    ):
        """Check the settings that do not depend on the problem, then store them.

        Raises:
            ConfigurationError: If ``hessian``, ``inner``, ``x_scale``,
                ``max_steps``, ``max_evaluations``, ``target_accepted_steps``,
                the radii, ``eta``, ``stationarity_tol``,
                ``max_consecutive_invalid_trials``, ``noise_trial_limit``,
                ``noise_relaxed_ratio``, ``gradient_floor``, ``weight_priors``,
                ``inner_start_iterations``, ``inner_fraction`` or
                ``inner_retreats`` are out of range.
        """
        if hessian not in ("gauss-newton", "true"):
            raise ConfigurationError(
                f"TrustRegion hessian must be 'gauss-newton' or 'true'; got {hessian!r}."
            )
        if inner is not None and (
            not isinstance(inner, TrustRegion) or inner.nested_at is not None
        ):
            raise ConfigurationError(
                f"TrustRegion inner= takes a TrustRegion without nested_at=; got {inner!r}."
            )
        x_scale = HessianScale() if x_scale is None else x_scale
        if not isinstance(x_scale, (HessianScale, JacScale, ConstantScale)):
            raise ConfigurationError(
                "TrustRegion x_scale must be ds.inference.HessianScale, ds.inference.JacScale, "
                "or ds.inference.ConstantScale."
            )
        if hessian == "true" and isinstance(x_scale, JacScale):
            raise ConfigurationError(
                "TrustRegion cannot use JacScale with hessian='true'. Use "
                "ds.inference.HessianScale() or ds.inference.ConstantScale(...)."
            )
        step_budget(max_steps, label=_LABEL)
        if target_accepted_steps is not None and int(target_accepted_steps) < 0:
            raise ConfigurationError(
                "TrustRegion target_accepted_steps must be nonnegative or None."
            )
        if float(radius_init) <= 0.0 or float(radius_max) < float(radius_init):
            raise ConfigurationError("TrustRegion requires 0 < radius_init <= radius_max.")
        if not 0.0 <= float(eta) < 1.0:
            raise ConfigurationError("TrustRegion eta must satisfy 0 <= eta < 1.")
        try:
            tolerance = 0.0 if stationarity_tol is None else float(stationarity_tol)
        except (TypeError, ValueError):
            tolerance = float("nan")
        if not tolerance >= 0.0:
            raise ConfigurationError(
                "TrustRegion stationarity_tol must be None or a number of at least 0; got "
                f"{stationarity_tol!r}."
            )
        if (
            isinstance(max_consecutive_invalid_trials, bool)
            or not isinstance(max_consecutive_invalid_trials, (int, np.integer))
            or max_consecutive_invalid_trials < 1
        ):
            raise ConfigurationError(
                "TrustRegion max_consecutive_invalid_trials must be an integer of at least 1; "
                f"got {max_consecutive_invalid_trials!r}."
            )
        if noise_trial_limit is not None and (
            isinstance(noise_trial_limit, bool)
            or not isinstance(noise_trial_limit, (int, float, np.integer, np.floating))
            or not float(noise_trial_limit) > 0.0
        ):
            raise ConfigurationError(
                "TrustRegion noise_trial_limit must be None or a positive number; got "
                f"{noise_trial_limit!r}."
            )
        if not isinstance(noise_relaxed_ratio, (bool, np.bool_)):
            raise ConfigurationError(
                "TrustRegion noise_relaxed_ratio must be True or False; got "
                f"{noise_relaxed_ratio!r}."
            )
        if not (
            gradient_floor is None
            or (isinstance(gradient_floor, str) and gradient_floor == "auto")
            or (
                not isinstance(gradient_floor, (bool, np.bool_))
                and isinstance(gradient_floor, (int, float, np.integer, np.floating))
                and bool(np.isfinite(gradient_floor))
                and float(gradient_floor) > 0.0
            )
        ):
            raise ConfigurationError(
                "TrustRegion gradient_floor must be 'auto', None or a positive number, the "
                f"relative accuracy of the lanes' derivatives; got {gradient_floor!r}."
            )
        if not isinstance(weight_priors, (bool, np.bool_)):
            raise ConfigurationError(
                f"TrustRegion weight_priors must be True or False; got {weight_priors!r}."
            )
        if max_evaluations is not None and (
            isinstance(max_evaluations, (bool, np.bool_))
            or not isinstance(max_evaluations, (int, np.integer))
            or max_evaluations < 1
        ):
            raise ConfigurationError(
                "TrustRegion max_evaluations must be None or an integer of at least 1; got "
                f"{max_evaluations!r}."
            )
        if (
            isinstance(inner_start_iterations, (bool, np.bool_))
            or not isinstance(inner_start_iterations, (int, np.integer))
            or inner_start_iterations < 0
        ):
            raise ConfigurationError(
                "TrustRegion inner_start_iterations must be an integer of at least 0; got "
                f"{inner_start_iterations!r}."
            )
        if isinstance(inner_fraction, (bool, np.bool_)) or not isinstance(
            inner_fraction, (int, float, np.integer, np.floating)
        ):
            fraction = float("nan")
        else:
            fraction = float(inner_fraction)
        if not 0.0 <= fraction < 1.0:
            raise ConfigurationError(
                "TrustRegion inner_fraction must be a number of at least 0 and below 1; got "
                f"{inner_fraction!r}."
            )
        if (
            isinstance(inner_retreats, (bool, np.bool_))
            or not isinstance(inner_retreats, (int, np.integer))
            or inner_retreats < 0
        ):
            raise ConfigurationError(
                "TrustRegion inner_retreats must be an integer of at least 0; got "
                f"{inner_retreats!r}."
            )
        self.hessian = hessian
        self.nested_at = nested_at
        self.inner = inner
        self.radius_init = radius_init
        self.radius_max = radius_max
        self.eta = eta
        self.grad_tol = grad_tol
        self.step_tol = step_tol
        self.cost_tol = cost_tol
        self.cost_atol = cost_atol
        self.stationarity_tol = stationarity_tol
        self.max_consecutive_invalid_trials = int(max_consecutive_invalid_trials)
        self.noise_trial_limit = None if noise_trial_limit is None else float(noise_trial_limit)
        self.noise_relaxed_ratio = bool(noise_relaxed_ratio)
        self.gradient_floor = (
            gradient_floor
            if gradient_floor is None or isinstance(gradient_floor, str)
            else float(gradient_floor)
        )
        self.max_steps = max_steps
        self.max_evaluations = None if max_evaluations is None else int(max_evaluations)
        self.inner_start_iterations = int(inner_start_iterations)
        self.inner_fraction = fraction
        self.inner_retreats = int(inner_retreats)
        self.target_accepted_steps = target_accepted_steps
        self.x_scale = x_scale
        self.weights = weights
        self.weight_priors = bool(weight_priors)
        self.constraints = tuple(constraints)
        self.gauges = tuple(gauges)
        self.fixed_indices = tuple(fixed_indices)
        self.callbacks = tuple(callbacks)

    def solve(self, problem: Any, *, callbacks: tuple = (), state: Any = None) -> OptResult:
        """Fit ``problem``.

        Args:
            problem: The optimization problem of a lowered study, whose
                structure holds ``"batched_lanes"`` (what ``optsolve`` passes),
                or the lowered study itself (``ds.lower(study)``), which is
                lowered here.
            callbacks: Callbacks of this fit, after the solver's own.
            state: The state a step record or a saved result carries, to
                continue that fit: the point, the radius, the growth lock,
                the scale, the gradient threshold, the counters and the
                histories (in the nested mode also the members' radii, locks
                and gradient thresholds and the polish's switch) are
                restored, the blocks are evaluated once at the
                point, and the next attempt is the one the uninterrupted fit
                would make; ``max_steps`` counts the attempts of the whole
                fit. ``None`` starts afresh.

        Returns:
            The [OptResult][diff_soup.inference.OptResult]; on the lanes it
            carries ``work`` and ``state``.

        Raises:
            ConfigurationError: If the settings do not fit the problem, or
                ``hessian="true"`` is asked of a study whose plan does not
                offer the lanes' ``hessian`` mode.
            ConfigurationError: If ``problem`` is not evaluated on the lanes
                of a lowered study (its structure lacks ``"batched_lanes"``).
            TypeError: If ``problem`` is neither an optimization problem nor
                a lowered study.
            ValueError: If a lane that no rung solves fails at the start
                point or at an accepted point (a resumed point included), the
                start point fails the trial predicate, or ``state`` was saved
                by another solver, mode, cut level, problem or number of
                coordinates.
        """
        callbacks = tuple(callbacks)
        problem = _lowered(self, problem)
        # The modes the problem's lanes evaluate decide the method: the blocks give
        # Gauss-Newton steps, the hessian mode the exact Hessian's.
        offers = problem.lanes._offers
        if self.hessian == "true":
            if self.nested_at is not None:
                raise ConfigurationError(
                    "TrustRegion nested_at= runs on the lanes of a lowered study with "
                    "hessian='gauss-newton', and this solver has hessian='true'; lower the "
                    "study with ds.lower(study) and use hessian='gauss-newton'."
                )
            if offers("value") and not offers("hessian"):
                raise ConfigurationError(
                    "TrustRegion hessian='true' steps on the exact Hessian of the lanes' "
                    "hessian mode, which this lowered study does not offer: the adjoints of an "
                    "experiment's solver cannot differentiate its lanes' reverse gradient "
                    "forward; use hessian='gauss-newton'."
                )
        route = _LaneRoute(self, problem)
        if self.nested_at is not None:
            route.refuse_inner_constraints(self.nested_at)
        if state is not None:
            _check_state(state, self, route)
            # The lanes' measured costs continue from the saved ledger, in the fit's own.
            ledger = getattr(route.problem, "_ledger", None)
            if ledger is not None:
                ledger.restore(state)
        if self.nested_at is not None:
            return _NestedFit(self, route, callbacks).run(state)
        return _JointFit(self, route, callbacks).run(state)


# ---------------------------------------------------------------------------
# The joint attempt
# ---------------------------------------------------------------------------


class _JointFit:
    """One trust region over every coordinate of the levels."""

    # The subproblem the result's ``info`` names.
    subproblem = "level-exact"

    def __init__(self, solver: TrustRegion, route: _LaneRoute, callbacks: tuple):
        self.solver = solver
        self.route = route
        self.callbacks = tuple(solver.callbacks) + callbacks if is_primary() else ()
        self.work = _Work(route)
        # The mode whose matrix the attempts step on: the Gauss-Newton blocks, or the
        # exact Hessian under hessian="true".
        self.matrix = "hessian" if solver.hessian == "true" else "blocks"
        # The bound of the gradient's error at the current point, from the lanes'
        # tolerances and blocks.
        self.floor = _GradientFloor(solver, route)

    def evaluate(self, mode: str, point: np.ndarray, lanes: Any = None) -> Any:
        """One request at ``point``, every lane (or ``lanes``) at the rung it holds.

        A lane the evaluation solved at a higher rung holds that rung from then on.
        A request of the matrix's mode (``blocks``, or ``hessian`` under
        ``hessian="true"``) counts in the work's ``blocks``, any other in ``value``.
        """
        results = self.route.evaluate(mode, point, **self.work.request(lanes))
        self.work.hold(results)
        if mode == self.matrix:
            self.work.blocks += 1
        else:
            self.work.value += 1
        return results

    def resolve(
        self, point: np.ndarray, raised: np.ndarray, before: np.ndarray, lane_costs: np.ndarray
    ) -> tuple:
        """The consistency re-solve after the lanes ``raised`` moved to a higher rung at a trial.

        One ``value`` request at the current point ``point`` over those lanes,
        each at its new rung. Returns the current lane costs with theirs
        replaced by the re-solved ones, and the change of the weighted cost;
        a lane that does not solve there keeps its cost. Each lane's switch is
        recorded with its old rung ``before``, its new rung, and its cost at the
        current point before and after.
        """
        route, work = self.route, self.work
        positions = np.sort(np.asarray(raised, dtype=np.int64))
        lanes = tuple(route.ids[position] for position in positions)
        results = self.evaluate("value", point, lanes)
        work.rung_switch_resolves += 1
        failed = {failure.lane for failure in route.failures(results)}
        costs = np.asarray(results.cost, dtype=np.float64)[0]
        replaced = np.array(lane_costs, dtype=np.float64, copy=True)
        change = 0.0
        for column, position in enumerate(positions):
            old = float(replaced[position])
            new = float("nan") if lanes[column] in failed else float(costs[column])
            if np.isfinite(new):
                replaced[position] = new
                change += float(route.omega[position]) * (new - old)
            work.attempt_switches.append(
                (lanes[column], int(before[position]), int(work.lane_rungs[position]), old, new)
            )
        return replaced, change

    def blocks(self, point: np.ndarray, *, start: bool) -> tuple:
        """The system at an accepted point (or the start), every lane solved there.

        Every lane of nonzero weight, that is: a failed lane of weight zero is
        logged as ignored. Returns the system and the lane costs at ``point``.
        """
        route = self.route
        results = self.evaluate(self.matrix, point)
        failures, ignored = route.weigh(route.failures(results))
        if failures and start:
            raise ValueError(
                f"the initial solve failed for lane(s) {_failures_text(failures)}; every lane "
                "has to solve at the start point."
            )
        if failures:
            raise ValueError(
                f"lane(s) {_failures_text(failures)} failed at an accepted point in the "
                f"{self.matrix} mode; every lane has to solve at an accepted point."
            )
        self.work.record(self.work.attempts, ignored, inner=False)
        system = route.system(results, point)
        self.work.held = system.held
        # The point becomes the current one: the gradient floor's bound reads its lanes.
        self.floor.current = self.floor.record(results)
        return system, np.array(results.cost, dtype=np.float64)[0]

    def state(self, loop: LoopState, **nested: Any) -> MappingProxyType:
        """The state after an attempt, read-only host arrays; ``nested``: the nested mode's own."""
        route = self.route
        entries = {
            "solver": np.asarray(_LABEL),
            "mode": np.asarray("joint"),
            "nested_at": np.asarray(""),
            "fingerprint": np.asarray(route.fingerprint),
            "n": np.asarray(route.n, dtype=np.int64),
            **loop.entries(),
        }
        entries.update(self.nested_entries(nested))
        entries.update({f"work.{key}": value for key, value in self.work.mapping().items()})
        # What the lanes cost so far, and the plan they ran under; never where they ran.
        problem = route.problem
        ledger = getattr(problem, "_ledger", None)
        if ledger is not None:
            entries.update(ledger.entries())
        fingerprint = getattr(problem, "_plan_fingerprint", None)
        value = fingerprint() if callable(fingerprint) else None
        if value is not None:
            entries["ledger.plan_fingerprint"] = np.asarray(value)
        return MappingProxyType(entries)

    def nested_entries(self, values: dict) -> dict:
        """The entries of the state that only the nested mode has (none here)."""
        return {}

    def member_above(
        self, members: Any, noise: float, tolerance: Any = None, floors: Any = None
    ) -> Any:
        """The member whose model predicts the most beyond the members' yardstick (none here)."""
        return None

    def verdict(
        self,
        stop: str,
        *,
        sigma: float,
        cost_tol: float,
        lock: int,
        invalid_run: int,
        radius: float,
        windows: tuple,
        members: Any = None,
        floor: Any = None,
    ) -> tuple:
        """The status, the message and the ``info`` entries of the fit's end.

        ``stop`` is the event that ended the loop and ``sigma`` the stationarity
        number at the end point; ``windows`` holds the noise-dominated trials'
        cost changes and the rejected trials' cost changes and predicted
        decreases; ``members`` the members' stationarity numbers at the end
        point in the nested mode; ``floor`` the gradient floor there
        (``_FloorEnd``). A fit that converged to the accuracy of its
        derivatives says so in its message, with its numbers, and so does a
        fit that converged at the jumps of its computed cost.
        """
        noise_changes, rejected_changes, rejected_decreases = windows
        noise = _median(noise_changes)
        floors = None if floor is None else floor.members
        above = self.member_above(members, noise, floors=floors)
        floor_sigma = 0.0 if floor is None else floor.sigma
        # With a gradient floor, the jumps of the computed cost that every rejected trial
        # met, when they all did, are what the verdict reads beside the measured noise.
        ceiling = _noise_ceiling(self.route)
        jumps = None
        if self.floor.coefficients is not None:
            jumps = _jump_level(rejected_changes, rejected_decreases, ceiling)
        jump_yardstick = None if jumps is None else max(cost_tol, 3.0 * jumps)
        jump_above = None if jumps is None else self.member_above(members, jumps, floors=floors)
        yardstick = max(cost_tol, 3.0 * noise)
        status, stall_reason = _verdict(
            stop,
            sigma=sigma,
            yardstick=yardstick,
            lock=lock,
            members=above is None,
            floor_sigma=floor_sigma,
            floor_known=floor is not None and floor.level is not None,
            jump_yardstick=jump_yardstick,
            jump_members=jump_above is None,
        )
        at_jumps = (
            jump_yardstick is not None
            and status in (TerminationStatus.STEP_TOL, TerminationStatus.COST_TOL)
            and not (sigma <= yardstick and above is None)
        )
        failures = _last_invalid(self.work.log)
        stalled = None
        if status is TerminationStatus.STALLED:
            stalled = _stalled_message(
                stall_reason,
                stop,
                consecutive=invalid_run,
                failures=failures,
                radius=radius,
                sigma=sigma,
                noise=noise,
                ceiling=_noise_ceiling(self.route),
                member=above,
            )
        budget = None
        if stop == "max_evaluations":
            budget = (
                f"{self.work.blocks + self.work.value} evaluations reached "
                f"max_evaluations={self.solver.max_evaluations}"
            )
        message, report = _stop_report(
            status, rejected_changes, rejected_decreases, stalled, budget
        )
        if status is TerminationStatus.DERIVATIVE_ACCURACY:
            message = self.floor.message(sigma, floor_sigma)
        if at_jumps:
            message = (
                f"{status.value}: every rejected trial met a jump of the computed cost (median "
                f"{jumps:.2g}, within the noise {ceiling:.2g} the solver tolerance allows), and "
                f"the model's predicted decrease ({sigma:.2g}) is within {jump_yardstick:.2g}, "
                "the larger of cost_tol and three times that jump, all relative to the cost "
                "scale max(|cost|, cost_atol / cost_tol); no trial can show a decrease that "
                "small"
            )
            report = {"evaluation_jumps": jumps, "cost_floor": jump_yardstick}
        entries = {
            "stop": stop,
            "stall_reason": stall_reason,
            "consecutive_invalid_trials": int(invalid_run),
            "failed_lanes": tuple(
                (
                    failure.lane,
                    failure.kind,
                    float(failure.t_reached),
                    tuple(int(rung) for rung in failure.rungs),
                )
                for failure in failures
            ),
            **report,
        }
        return status, message, entries

    def step_info(self, loop: LoopState, attempt: int, **fields: Any) -> StepInfo:
        """The ``StepInfo`` of attempt ``attempt``: the loop's numbers, the work, and ``fields``."""
        work = self.work
        return StepInfo(
            step=attempt,
            cost=loop.cost,
            params=np.array(loop.point),
            wall_time=time.perf_counter() - self.started,
            radius=float(loop.radius),
            accepted_count=loop.accepted,
            attempt_count=attempt + 1,
            residual_evals=work.blocks + work.value,
            jacobian_evals=work.blocks,
            trial_cost_evals=work.value,
            statistics_evals=work.blocks,
            trial_predicate_rejections=work.predicate_rejections,
            invalid_pde_trials=work.invalid_trials,
            block_solves=work.solves,
            **fields,
        )

    def notify(self, info: StepInfo) -> bool:
        """Hand ``info`` to every callback; whether one asked, in any process, to stop the fit."""
        requested = False
        for callback in self.callbacks:
            try:
                callback.on_step(info)
            except StopFit:
                requested = True
        return any_process(requested)

    def outcome(
        self,
        loop: LoopState,
        stop: str,
        *,
        attempt: int,
        x0: np.ndarray,
        cost_tol: float,
        grad_norm: float,
        projected_grad_norm: float,
        stationarity: float,
        members: Any = None,
        floor: Any = None,
        **nested: Any,
    ) -> OptResult:
        """The fit's result, ended by ``stop`` after attempt ``attempt``, handed to the callbacks.

        ``stationarity`` is the stationarity number at the end point and
        ``members`` the members' numbers there in the nested mode; ``floor``
        the gradient floor there (``_FloorEnd``); ``nested`` holds the nested
        mode's own entries of the final state.
        """
        work, route = self.work, self.route
        loop.attempt = work.attempts
        state = self.state(loop, **nested)
        status, message, report = self.verdict(
            stop,
            sigma=stationarity,
            cost_tol=cost_tol,
            lock=loop.growth_lock,
            invalid_run=loop.consecutive_invalid_trials,
            radius=loop.radius,
            windows=(
                loop.noise_dominated_cost_change,
                loop.rejected_cost_change,
                loop.rejected_predicted_decrease,
            ),
            members=members,
            floor=floor,
        )
        evaluations = work.blocks + work.value
        result = OptResult(
            params=np.array(loop.point),
            initial_params=x0,
            cost=loop.cost,
            grad_norm=grad_norm,
            projected_grad_norm=projected_grad_norm,
            status=status,
            message=message,
            n_steps=attempt + 1,
            max_steps=int(self.solver.max_steps),
            n_accepted=loop.accepted,
            eval_counts=EvalCounts(
                residual_evals=evaluations,
                jacobian_evals=work.blocks,
                pde_solves=evaluations * len(route.ids),
                trial_cost_evals=work.value,
                statistics_evals=work.blocks,
                trial_predicate_rejections=work.predicate_rejections,
                invalid_pde_trials=work.invalid_trials,
                block_solves=work.solves,
            ),
            cost_history=np.asarray(loop.cost_history, dtype=np.float64),
            grad_history=np.asarray(loop.grad_history, dtype=np.float64),
            wall_time=time.perf_counter() - self.started,
            info={
                "failure_sentinel_absent": True,
                "hessian": "gauss-newton" if self.matrix == "blocks" else "true",
                "hessian_composition": (
                    None if self.matrix == "blocks" else "forward-over-reverse"
                ),
                "linear_solver": "cholesky",
                "subproblem": self.subproblem,
                "active_bounds": box.active_bounds(loop.point, route.bounds),
            },
            work=work.mapping(),
            state=state,
            trust_region=TrustRegionReport(
                **{
                    "radius_history": tuple(loop.radius_history),
                    "stationarity": stationarity,
                    "member_stationarity": (
                        None if members is None else tuple(float(value) for value in members)
                    ),
                    "evaluation_noise": _median(loop.noise_dominated_cost_change),
                    "evaluation_noise_trials": len(loop.noise_dominated_cost_change),
                    "evaluation_noise_ceiling": _noise_ceiling(route),
                    **({} if floor is None else floor.entries),
                    **report,
                }
            ),
        )
        for callback in self.callbacks:
            callback.on_end(result)
        return result

    def floor_end(
        self,
        loop: LoopState,
        step: _FullStep,
        scale: Any,
        cost_tol: float,
        relative_to: float,
        member_steps: Any = None,
        system: Any = None,
    ) -> _FloorEnd:
        """The gradient floor at the end point: what the verdict reads, and the ``info`` entries.

        ``step`` is the model's full step at the end point and ``scale`` the
        coordinates the bound reads there; ``relative_to`` is the cost scale
        there (``cost_scale``); ``member_steps`` holds every member's
        full-step norm over its cost scale in the nested mode; ``system``,
        when given, is the full system at the end point, whose dense matrix
        gives the floor's reach along its softest direction. Before the first
        floor sample the bound stands in for the floor in the entries, and no
        verdict reads it.
        """
        samples = loop.gradient_floor_samples
        bound = self.floor.bound(scale)
        level = _floor_level(samples, bound)
        reported = bound if level is None else level
        source = None
        if bound is not None:
            source = "a_posteriori" if samples and _median(samples) <= bound else "a_priori"
        radius = _floor_parameter_radius(reported, step.norm, step.length, step.decrease)
        box = None
        if reported is not None and system is not None:
            box = float("nan")
            if step.norm > 0.0 and step.length > 0.0:
                # The floor per unit parameter length along the full step, over the
                # smallest eigenvalue of the model's matrix: how far the minimizer can
                # move along the softest direction within the gradient's accuracy.
                matrix, _ = elimination.level_dense(system)
                smallest = float(np.linalg.eigvalsh(np.asarray(matrix, dtype=np.float64))[0])
                if smallest > 0.0:
                    box = reported * step.norm / step.length / smallest
        entries = {
            "gradient_floor": (
                None if reported is None else float(_floor_sigma(reported, step.norm, relative_to))
            ),
            "gradient_floor_bound": (
                None if bound is None else float(_floor_sigma(bound, step.norm, relative_to))
            ),
            "gradient_floor_trials": len(samples),
            "cost_floor": max(float(cost_tol), 3.0 * _median(loop.noise_dominated_cost_change)),
            "floor_parameter_radius": None if radius is None else float(radius),
            "floor_parameter_box": None if box is None else float(box),
            "floor_source": source,
        }
        members = (
            None
            if level is None or member_steps is None
            else level * np.asarray(member_steps, dtype=np.float64)
        )
        return _FloorEnd(
            level, float(_floor_sigma(level, step.norm, relative_to)), members, entries
        )

    def run(self, state: Any = None) -> OptResult:
        """Fit from the start point, or continue the fit whose ``state`` is given."""
        solver, route, work = self.solver, self.route, self.work
        dtype = np.asarray(route.problem.x0).dtype
        x0 = np.array(route.problem.x0, dtype=dtype)
        bounds = route.bounds
        if state is None:
            x = x0.copy()
            if bounds is not None:
                # Iterates stay strictly inside the box: a start on or beyond a side
                # moves to the next float inside it.
                x = box.make_strictly_feasible(x, bounds[0], bounds[1], rstep=0)
            if not trial_is_valid(route.view, x):
                raise ValueError("TrustRegion initial point failed the trial predicate.")
        else:
            # The saved point is the start or an accepted point: every lane solved
            # there and the predicate passed. Its blocks are evaluated again below.
            x = np.array(state["point"], dtype=dtype)
            work.restore(state)
            work.resume_blocks += 1
        # grad_tol=None tests no gradient: the stationarity number is the test there.
        grad_tolerance, step_tol, cost_tol, cost_atol, _ = loop_tolerances(
            dtype,
            grad_tol=solver.grad_tol,
            step_tol=solver.step_tol,
            cost_tol=solver.cost_tol,
            cost_atol=solver.cost_atol,
            stationarity_tol=solver.stationarity_tol,
            label=_LABEL,
        )
        fixed = np.asarray(route.fixed, dtype=np.int64)
        n_constraints = int(route.constraint_matrix.shape[0])
        route.problem.prepare(modes=("value", self.matrix))
        system, lane_costs = self.blocks(x, start=state is None)
        cost = float(system.cost)
        # Every program the fit runs is compiled here, before the loop: the scale, the
        # attempt's program (the solve at lambda = 0 and the step of the initial radius,
        # kept as the first attempt's of a fresh fit), the quadratic, and the solve at
        # lambda = 0 alone that the end point's numbers take. The regularizer was
        # evaluated at the point by the system's blocks.
        gradient, diagonal = route.vectors(system)
        start = _subproblem(x, gradient, _scale_candidate(system, solver.x_scale, diagonal), bounds)
        first = route.solve_attempt(system, start, float(solver.radius_init))
        elimination.level_quadratic(system, np.zeros_like(x))
        elimination.solve_scaled(system, start.scale, 0.0, start.addition)

        for callback in self.callbacks:
            callback.on_start(route.n, route.view.n_residuals, solver.max_steps)
        self.started = time.perf_counter()
        if state is None:
            # A fresh fit's first record holds the retries of the start's request.
            record = _record(
                float("nan"), float("nan"), False, x, (), route.depth, **work.attempt_records()
            )
            loop = LoopState.start(
                point=x, cost=cost, scale=None, radius=float(solver.radius_init), record=record
            )
        else:
            # A resumed fit's first record holds the retries of its first attempt, which
            # makes its own lambda = 0 solve in the restored scale, as an uninterrupted fit.
            first = None
            loop = LoopState.restore(
                state, point=x, cost=cost, scale=np.array(state["scale"], dtype=dtype)
            )
        # The largest relative predicted decrease of a trial counted as noise-dominated;
        # None bounds none.
        noise_limit = _noise_trial_limit(solver, route)
        start_attempt = loop.attempt
        attempt = start_attempt
        for attempt in range(start_attempt, int(solver.max_steps)):
            gradient, diagonal = route.vectors(system)
            candidate = _scale_candidate(system, solver.x_scale, diagonal)
            loop.scale = candidate if loop.scale is None else np.maximum(loop.scale, candidate)
            grad_norm = float(np.linalg.norm(gradient))
            # The subproblem's coordinates: the scale, under bounds shaped by the
            # distance to the sides the gradient points at.
            coordinates = _subproblem(loop.point, gradient, loop.scale, bounds)
            stop_gradient = float(np.linalg.norm(_bounded_gradient(gradient, coordinates, bounds)))
            loop.cost_history.append(loop.cost)
            loop.grad_history.append(grad_norm)
            if grad_tolerance is not None:
                if loop.grad_threshold is None:
                    loop.grad_threshold = grad_tolerance.threshold(stop_gradient)
                if not n_constraints and stop_gradient <= loop.grad_threshold:
                    stop = "gradient_tolerance"
                    break

            # The solve at lambda = 0 in this attempt's scale gives the stationarity
            # number of the current point and is the first solve of the step's
            # multiplier search; one program makes both, with the step of this radius
            # (a fresh fit's first attempt has them from above).
            if (
                attempt == start_attempt
                and first is not None
                and first.answers(coordinates.scale, loop.radius, coordinates.addition)
            ):
                zero = first
            else:
                zero = route.solve_attempt(system, coordinates, loop.radius)
            # Every relative number of the attempt is relative to the cost scale here.
            relative_to = cost_scale(loop.cost, cost_tol, cost_atol)
            stationarity = _stationarity(
                system, gradient, coordinates.scale, zero, relative_to, coordinates.addition
            )
            # The model predicts no further decrease the evaluation can show here.
            if stationarity <= _stationarity_limit(
                solver.stationarity_tol, cost_tol, _median(loop.noise_dominated_cost_change)
            ):
                work.solves += 1
                stop = "stationarity"
                break
            # The gradient's accuracy floor here: the median of the floor samples, at
            # most the bound from the lanes' tolerances and blocks; None before the
            # first sample, when no rule reads it.
            bound = self.floor.bound(coordinates.scale)
            level = _floor_level(loop.gradient_floor_samples, bound)
            full_norm = float(np.linalg.norm(np.asarray(zero[0])))
            # The model predicts no further decrease its derivatives resolve here.
            if level is not None and stationarity <= _floor_sigma(level, full_norm, relative_to):
                work.solves += 1
                stop = "derivative_accuracy"
                break
            radius_floor = _radius_floor(level, full_norm, gradient, coordinates.scale)
            step_radius = loop.radius
            trial, scaled_step_norm, predicted, multiplier, solves = _trial_point(
                system,
                loop.point,
                gradient,
                coordinates,
                loop.radius,
                first=zero,
                fixed=fixed,
                bounds=bounds,
                equality=n_constraints > 0,
            )
            work.solves += solves
            step = trial - loop.point
            step_norm = float(np.linalg.norm(step))

            trial_ok = _trial_valid(route.view, trial)
            failures: tuple = ()
            failed_owners = None
            rejection_reason = None
            invalid = False
            if not trial_ok:
                work.predicate_rejections += 1
                trial_cost = float("inf")
                rejection_reason = "the trial predicate rejected the point"
            else:
                before = work.lane_rungs.copy()
                results = self.evaluate("value", trial)
                # A failed lane of weight zero is outside the objective: logged, ignored.
                failures, ignored = route.weigh(route.failures(results))
                work.record(attempt, ignored, inner=False)
                raised = np.flatnonzero(work.lane_rungs > before)
                if raised.size:
                    # A lane that moved to a higher rung at the trial is solved again at
                    # the current point at its new rung, so that the gain ratio compares
                    # costs of the same rung; the model there stays the earlier rung's.
                    lane_costs, change = self.resolve(loop.point, raised, before, lane_costs)
                    loop.cost += change
                if failures:
                    invalid = True
                    work.invalid_trials += 1
                    work.record(attempt, failures, inner=False)
                    trial_cost = float("inf")
                    failed_owners = route.owners(failures)
                    rejection_reason = (
                        "the lane solve failed at the trial for experiment(s) "
                        f"{_listed(failure.lane for failure in failures)} "
                        f"({_listed(failure.kind for failure in failures)})"
                    )
                else:
                    trial_cost = route.trial_cost(results, trial)

            predicted_value = float(predicted)
            rho, accepted, loop.radius, reason = _judge(
                cost=loop.cost,
                trial_cost=trial_cost,
                predicted=predicted_value,
                valid=trial_ok,
                invalid=invalid,
                scaled_norm=scaled_step_norm,
                radius=loop.radius,
                lock=loop.growth_lock,
                noise_window=loop.noise_dominated_cost_change,
                eta=float(solver.eta),
                radius_max=solver.radius_max,
                relaxed=solver.noise_relaxed_ratio,
                relative_to=cost_scale(loop.cost, cost_tol, cost_atol),
                radius_floor=radius_floor,
            )
            assert_decision(accepted)
            rejection_reason = rejection_reason or reason

            previous_cost = loop.cost
            if accepted:
                loop.point = trial
                system, lane_costs = self.blocks(loop.point, start=False)
                loop.cost = float(system.cost)
                loop.accepted += 1
            work.attempted(accepted)
            loop.radius_history.append(loop.radius)
            after = _after_attempt(
                loop,
                accepted=accepted,
                valid=trial_ok,
                invalid=invalid,
                cost=previous_cost,
                trial_cost=trial_cost,
                predicted=predicted_value,
                step_radius=step_radius,
                cost_tol=cost_tol,
                cost_atol=cost_atol,
                noise_limit=noise_limit,
                scaled_norm=scaled_step_norm,
                floor_bound=bound,
            )
            loop = dataclasses.replace(loop, **after.trials._asdict())
            loop.record = _record(
                rho,
                float(multiplier),
                accepted,
                trial,
                failures,
                route.depth,
                **work.attempt_records(),
            )
            loop.attempt = attempt + 1
            info = self.step_info(
                loop,
                attempt,
                grad_norm=grad_norm,
                accepted=accepted,
                delta_norm=step_norm,
                rho=rho,
                lam=float(multiplier),
                rejection_reason=rejection_reason,
                failed_owners=failed_owners,
                state=self.state(loop),
                stationarity=stationarity,
            )
            stop = _stop_after(
                after,
                requested=self.notify(info),
                max_invalid=int(solver.max_consecutive_invalid_trials),
                # The step test reads every trial whose cost was computed, at the point
                # where the acceptance left the fit.
                step_small=step_small(step_norm, loop.point, step_tol, tested=after.evaluated),
                accepted_count=loop.accepted,
                target_steps=solver.target_accepted_steps,
                # The budget counts the consistency re-solves too.
                evaluations=work.blocks + work.value,
                max_evaluations=solver.max_evaluations,
            )
            if stop is not None:
                break
        else:
            attempt = max(int(solver.max_steps), start_attempt) - 1
            stop = "max_steps"

        gradient, diagonal = elimination.level_vectors(system)
        grad_norm = float(np.linalg.norm(gradient))
        # The stationarity number at the end point, in the coordinates the next
        # attempt would take there.
        candidate = _scale_candidate(system, solver.x_scale, diagonal)
        end_scale = candidate if loop.scale is None else np.maximum(loop.scale, candidate)
        end = _subproblem(loop.point, gradient, end_scale, bounds)
        projected_grad_norm = float(np.linalg.norm(_bounded_gradient(gradient, end, bounds)))
        end_step = _full_step(
            system,
            gradient,
            end.scale,
            elimination.solve_scaled(system, end.scale, 0.0, end.addition),
            end.addition,
        )
        end_relative_to = cost_scale(loop.cost, cost_tol, cost_atol)
        end_stationarity = end_step.decrease / end_relative_to
        return self.outcome(
            loop,
            stop,
            attempt=attempt,
            x0=x0,
            cost_tol=cost_tol,
            grad_norm=grad_norm,
            projected_grad_norm=projected_grad_norm,
            stationarity=end_stationarity,
            floor=self.floor_end(
                loop, end_step, end.scale, cost_tol, end_relative_to, system=system
            ),
        )


# ---------------------------------------------------------------------------
# The nested attempt
# ---------------------------------------------------------------------------


# Why a member's polish ended at an attempt: the codes of the record's ``inner_stop``.
_NOT_POLISHED = 0  # no polish: the outer trial was not evaluated, or the member holds no data
_ACCURACY = 1  # its predicted decrease fell to inner_fraction of the outer step's
_AT_MAX_STEPS = 2  # it made the polish's largest number of iterations
_STEP_TOLERANCE = 3  # its step test
_COST_TOLERANCE = 4  # its cost test
_GRADIENT_TOLERANCE = 5  # its gradient test
_STATIONARITY = 6  # its stationarity number at most the inner stationarity_tol
_EVALUATION_NOISE = 7  # three noise-dominated rejected trials in a row
_RADIUS = 8  # its radius fell to the smallest positive float
_SKIPPED = 9  # the polish was switched off at this attempt
_NO_FINITE_STEP = 10  # no finite step inside its radius: its subtree's model is not finite

# The polish's iterations per member when the nested solver is given no inner solver.
_DEFAULT_POLISH = 2
# A polish whose gain is below this fraction of the outer step's predicted decrease on
# _SMALL_GAINS accepted attempts in a row is switched off until an outer rejection.
_POLISH_GAIN = 1e-3
_SMALL_GAINS = 2


class _InnerResult(NamedTuple):
    """An inner solve at one outer point: where it ended, its system, and the inner state.

    ``failures`` holds the failed lanes of a member that no retreat solved at
    an outer trial, whose member cannot be solved there; ``system`` is then
    ``None`` and ``point`` the last point tried. ``stationarity`` holds every
    member's stationarity number at ``point``, ``decreases`` the decrease its
    model predicts for its full step there, ``lane_costs`` the lane costs
    there, ``codes`` why every member's iterations ended (the codes of
    ``inner_stop``) and ``first_cost`` the cost at the first request (all
    ``None`` when the solve ended at a failure). ``retreats`` holds every
    member's retreats in the solve. ``full_steps`` holds every member's
    full-step norm over its cost at ``point`` (``member_full_steps``) and
    ``floor_terms`` the lanes' terms of the gradient floor's bound there
    (both ``None`` when the solve ended at a failure).
    """

    point: np.ndarray
    system: Any
    scale: Any
    radii: np.ndarray
    locks: np.ndarray
    thresholds: np.ndarray
    iterations: np.ndarray
    failures: tuple
    stationarity: Any = None
    lane_costs: Any = None
    decreases: Any = None
    codes: Any = None
    first_cost: Any = None
    retreats: Any = None
    full_steps: Any = None
    floor_terms: Any = None


def _member_numbers(decreases: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Every member's predicted decrease over its cost scale ``scales``; ``nan`` stays ``nan``."""
    with np.errstate(all="ignore"):
        return np.asarray(decreases, dtype=np.float64) / np.asarray(scales, dtype=np.float64)


def _finite_blocks(system: Any) -> bool:
    """Whether every block of a system's matrix and its gradient are finite."""
    arrays = [system.root] + list(system.diagonal[1:])
    arrays += [entry for row in system.cross[1:] for entry in row]
    arrays += [system.root_gradient] + list(system.gradient[1:])
    return all(bool(np.all(np.isfinite(np.asarray(entry)))) for entry in arrays)


class _NestedFit(_JointFit):
    """Two loops cut at one level: the levels above it, and every member's subtree below.

    The outer trust region steps on the Schur complement of the inner blocks
    into the outer blocks, with the outer gradient corrected for the gradient
    the members leave, the inner gradients folded through the inner factors;
    its gradient test (with ``grad_tol`` set) reads that corrected gradient.
    At an outer trial the members move by their response to the outer step,
    each bounded by its radius, then polish: every member of the cut level
    iterates on its own subtree with the outer coordinates held, all members
    in lockstep (one ``blocks`` and one ``value`` request over every lane per
    iteration), each with its own radius, growth lock and acceptance, for at
    most the polish's number of iterations. A member whose lane fails at its
    responded or polished point retreats toward its last point where its
    lanes solved. The trial cost is the cost after the polish. The members'
    radii and locks carry from one attempt to the next.
    """

    subproblem = "level-nested"

    def __init__(self, solver: TrustRegion, route: _LaneRoute, callbacks: tuple):
        super().__init__(solver, route, callbacks)
        self.inner = solver if solver.inner is None else solver.inner
        # The polish's largest number of iterations per member and attempt.
        self.polish_steps = _DEFAULT_POLISH if solver.inner is None else int(solver.inner.max_steps)
        self.cut = route.names.index(solver.nested_at) + 1
        counts, widths = route.tables.counts, route.tables.widths
        self.prefix = int(widths[0]) + sum(
            int(counts[level - 1]) * int(widths[level]) for level in range(1, self.cut)
        )
        self.members = tuple(route.members[self.cut - 1])
        self.lane_member = np.asarray(route.nodes[:, self.cut - 1], dtype=np.int64)
        # The member of the cut level of every joint column; set by the fit's first ``blocks``.
        self.owners: Any = None
        regularizer = route.view.regularization
        rows = regularizer.residual(jax.device_put(route.problem.x0, route.device))
        self.regularized = int(np.shape(rows)[0]) > 0
        self.work.nest(self.members)

    def nested_entries(self, values: dict) -> dict:
        """The mode, the cut level, every member's inner radius, lock and threshold, the switch.

        ``polish_off`` is 1 while the polish is switched off, and
        ``polish_small_gains`` the number of accepted attempts in a row whose
        polish made an iteration and gained too little.
        """
        return {
            "mode": np.asarray("nested"),
            "nested_at": np.asarray(str(self.solver.nested_at)),
            "inner_radius": np.array(values["inner_radius"], dtype=np.float64),
            "inner_lock": np.array(values["inner_lock"], dtype=np.int64),
            "inner_grad_threshold": np.array(values["inner_grad_threshold"], dtype=np.float64),
            "polish_off": np.asarray(int(values["polish_off"]), dtype=np.int64),
            "polish_small_gains": np.asarray(values["polish_small_gains"], dtype=np.int64),
        }

    def member_above(
        self, members: Any, noise: float, tolerance: Any = None, floors: Any = None
    ) -> Any:
        """The member whose model predicts the most beyond the members' limit, or ``None``.

        The limit is ``tolerance``, or for ``None`` the members' yardstick
        ``max(inner cost_tol, 3 * noise)``, ``noise`` the evaluation noise the
        outer trials measured; ``floors``, every member's gradient floor in
        its stationarity number's units, raises each member's limit to its
        floor (``None`` raises none). A member whose number is not finite is
        beyond it. Returns ``(level, label, number, limit)``.
        """
        if members is None:
            return None
        numbers = np.asarray(members, dtype=np.float64)
        limit = _stationarity_limit(tolerance, self.inner_tolerances.cost_tol, noise)
        limits = limit if floors is None else np.fmax(limit, np.asarray(floors, dtype=np.float64))
        beyond = np.flatnonzero(~(numbers <= limits))
        if not beyond.size:
            return None
        worst = beyond[np.argmax(np.where(np.isfinite(numbers[beyond]), numbers[beyond], np.inf))]
        if floors is not None:
            limit = float(limits[worst])
        return (self.solver.nested_at, self.members[int(worst)], float(numbers[worst]), limit)

    def members_stationary(self, members: Any, noise: float, floors: Any = None) -> bool:
        """Whether every member's stationarity number is within the inner stationarity limit.

        The limit is the inner ``stationarity_tol``, or for ``None`` the
        members' yardstick ``max(inner cost_tol, 3 * noise)``: the limit each
        member's own stationarity test reads in the polish; ``floors`` raises
        each member's limit to its gradient floor, as ``member_above``.
        """
        return (
            self.member_above(members, noise, self.inner_tolerances.stationarity_tol, floors)
            is None
        )

    def record(
        self,
        rho,
        lam,
        accepted,
        trial,
        failures,
        iterations,
        codes,
        *,
        retreats=None,
        capped=None,
        norms=None,
        owners=None,
        **attempt,
    ) -> dict:
        """The attempt just made, with every member's polish, retreats and response in it.

        Per member: its polish iterations, the code of why its polish ended,
        its retreats (``retreats``), whether its response was bounded by its
        radius (``capped``) and its response's scaled norm before the bound
        (``norms``); the last three are 0, ``False`` and ``nan`` when not
        given (the start's record). ``owners`` adds the split of the cost by
        member (``owner_split``) when given.
        """
        count = len(self.members)
        entries = _record(rho, lam, accepted, trial, failures, self.route.depth, **attempt)
        if owners is not None:
            entries.update(owners)
        entries["inner_iterations"] = np.array(iterations, dtype=np.int64)
        entries["inner_stop"] = np.array(codes, dtype=np.int64)
        entries["inner_retreats"] = (
            np.zeros((count,), dtype=np.int64)
            if retreats is None
            else np.array(retreats, dtype=np.int64)
        )
        entries["response_capped"] = (
            np.zeros((count,), dtype=bool) if capped is None else np.array(capped, dtype=bool)
        )
        entries["response_norms"] = (
            np.full((count,), np.nan) if norms is None else np.array(norms, dtype=np.float64)
        )
        return entries

    def valid(self, point: np.ndarray) -> bool:
        """The trial predicate at ``point``; a predicate that raises rejects it."""
        return _trial_valid(self.route.view, point)

    def member_costs(self, lane_costs: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Each member's cost: its lanes' weighted costs and the regularizer rows of its subtree.

        Every regularizer block counts with the route's weight of that block.
        """
        lanes = self.route.omega * np.asarray(lane_costs, dtype=np.float64)
        costs = np.bincount(self.lane_member, weights=lanes, minlength=len(self.members))
        if not self.regularized:
            return costs
        regularizer = self.route.view.regularization
        weights = self.route.block_weights
        declared = tuple(regularizer.block_rows(jax.device_put(point, self.route.device)))
        if weights is not None and len(declared) != len(weights):
            raise ValueError(
                f"the regularizer's block_rows(x) declares {len(declared)} blocks at this point "
                f"and {len(weights)} where its weights were attributed; a regularizer declares "
                "the same blocks at every point."
            )
        for index, (columns, residual, _) in enumerate(declared):
            members = self.owners[np.asarray(columns, dtype=np.int64)]
            below = members[members >= 0]
            if below.size:
                weight = None if weights is None else float(weights[index])
                if weight == 0.0:
                    continue
                residual = np.ravel(np.asarray(residual, dtype=np.float64))
                if weight is None:
                    costs[int(below[0])] += 0.5 * float(residual @ residual)
                else:
                    costs[int(below[0])] += weight * (0.5 * float(residual @ residual))
        return costs

    def shared_cost(self, point: np.ndarray) -> float:
        """The regularizer rows of no member: the blocks on coordinates above the cut level only."""
        if not self.regularized:
            return 0.0
        regularizer = self.route.view.regularization
        weights = self.route.block_weights
        declared = tuple(regularizer.block_rows(jax.device_put(point, self.route.device)))
        total = 0.0
        for index, (columns, residual, _) in enumerate(declared):
            members = self.owners[np.asarray(columns, dtype=np.int64)]
            if members[members >= 0].size:
                continue
            residual = np.ravel(np.asarray(residual, dtype=np.float64))
            weight = 1.0 if weights is None else float(weights[index])
            total += weight * (0.5 * float(residual @ residual))
        return total

    def owner_split(
        self, system: Any, point: np.ndarray, lane_costs: np.ndarray, solved: Any
    ) -> dict:
        """The attempt's cost split by member of the cut level, read off without changing the fit.

        ``owner_costs`` holds every member's cost at the current ``point``
        (``member_costs``) and ``owner_shared_cost`` the rest of the cost there
        (``shared_cost``); ``owner_trial_costs`` and
        ``owner_trial_shared_cost`` the same at the trial after its polish
        (``nan`` when the trial was not evaluated or a member's lanes did not
        solve there); ``owner_predicted`` every member's part of the decrease
        the Gauss-Newton model at ``point`` predicts for the whole move to the
        trial (its subtree's terms and their coupling to the outer step,
        ``model_split``) and ``owner_shared_predicted`` the outer coordinates'
        own terms; together they sum to the model's decrease for that move.
        """
        count = len(self.members)
        entries = {
            "owner_costs": np.asarray(self.member_costs(lane_costs, point), dtype=np.float64),
            "owner_shared_cost": np.asarray(self.shared_cost(point), dtype=np.float64),
            "owner_trial_costs": np.full((count,), np.nan),
            "owner_trial_shared_cost": np.asarray(np.nan),
            "owner_predicted": np.full((count,), np.nan),
            "owner_shared_predicted": np.asarray(np.nan),
        }
        if solved is None:
            return entries
        delta = np.asarray(solved.point, dtype=np.float64) - np.asarray(point, dtype=np.float64)
        members, rest = elimination.model_split(system, delta, self.cut)
        entries["owner_predicted"] = np.asarray(members, dtype=np.float64)
        entries["owner_shared_predicted"] = np.asarray(rest, dtype=np.float64)
        if not solved.failures and solved.lane_costs is not None:
            entries["owner_trial_costs"] = np.asarray(
                self.member_costs(solved.lane_costs, solved.point), dtype=np.float64
            )
            entries["owner_trial_shared_cost"] = np.asarray(
                self.shared_cost(solved.point), dtype=np.float64
            )
        return entries

    def member_decrease(
        self, system: Any, scale: Any, zero: tuple, addition: Any = None
    ) -> np.ndarray:
        """The decrease every member's model predicts for its full step at zero damping.

        ``zero`` is ``inner_solve_scaled(system, scale, cut, addition=addition)``;
        member ``n``'s full step is its part of ``z / scale``, and its decrease
        is the one its own model (the scaled matrix carrying ``addition`` on its
        diagonal) predicts for that step: ``nan`` for a member whose solve is
        not finite.
        """
        z, q_squared = np.asarray(zero[0]), np.asarray(zero[1], dtype=np.float64)
        with np.errstate(all="ignore"):
            delta = z / np.asarray(scale, dtype=z.dtype)
        owners = self.owners
        below = owners >= 0
        broken = np.bincount(
            owners[below],
            weights=(~np.isfinite(z[below])).astype(np.float64),
            minlength=len(self.members),
        )
        predicted = elimination.inner_predicted(system, delta, self.cut)
        if addition is not None:
            predicted -= 0.5 * elimination.inner_added(system, addition, z, self.cut)
        predicted[(broken > 0) | ~np.isfinite(q_squared)] = np.nan
        return predicted

    def member_stationarity(
        self, system: Any, scale: Any, zero: tuple, costs: np.ndarray, addition: Any = None
    ) -> np.ndarray:
        """Every member's stationarity number: ``member_decrease`` over its cost scale."""
        return _member_numbers(
            self.member_decrease(system, scale, zero, addition), self.member_scales(costs)
        )

    def member_scales(self, costs: Any) -> Any:
        """Every member's cost scale ``max(|cost_n|, cost_atol / cost_tol)``, inner tolerances."""
        tolerances = self.inner_tolerances
        return cost_scale(costs, tolerances.cost_tol, tolerances.cost_atol)

    def member_full_steps(self, zero: tuple, costs: np.ndarray) -> np.ndarray:
        """Every member's full-step norm over its cost scale (``member_scales``).

        ``zero`` is every member's solve at zero damping
        (``inner_solve_scaled``), in the scaled coordinates its radius is
        measured in. The gradient floor times this number is the member's
        floor in its stationarity number's units; ``nan`` for a member whose
        solve is not finite.
        """
        z = np.asarray(zero[0], dtype=np.float64)
        owners = self.owners
        below = owners >= 0
        with np.errstate(all="ignore"):
            norms = np.sqrt(
                np.bincount(owners[below], weights=z[below] ** 2, minlength=len(self.members))
            )
        return norms / self.member_scales(costs)

    def members_at(self, system: Any, scale: Any, point: np.ndarray, lane_costs: Any) -> tuple:
        """Every member's predicted decrease, stationarity number and full step at ``point``.

        From one solve at zero damping of every subtree, in the subproblem's
        coordinates at ``point`` with ``scale``. Returns ``(decreases,
        numbers, full_steps)``, ``full_steps`` as ``member_full_steps`` gives
        them.
        """
        gradient, _ = self.route.vectors(system)
        coordinates = _subproblem(point, gradient, scale, self.route.bounds)
        zero = elimination.inner_solve_scaled(
            system, coordinates.scale, self.cut, addition=coordinates.addition
        )
        decreases = self.member_decrease(system, coordinates.scale, zero, coordinates.addition)
        costs = self.member_costs(lane_costs, point)
        numbers = _member_numbers(decreases, self.member_scales(costs))
        return decreases, numbers, self.member_full_steps(zero, costs)

    def full_stationarity(
        self,
        system: Any,
        reduced: Any,
        point: np.ndarray,
        scale: np.ndarray,
        relative_to: float,
        decreases: np.ndarray,
        *,
        outer: Any = None,
        zero: Any = None,
    ) -> tuple:
        """The stationarity number of the full system at ``point``, the solves, the outer step.

        The decrease the full Gauss-Newton model predicts for its full step is
        the reduced model's for its full step plus every member's for its own
        (``decreases``, the outer coordinates held), since the reduced system
        is the Schur complement of the members' blocks; the number is that sum
        over ``relative_to``, the cost scale at the point (``cost_scale``), the
        joint mode's number at the same point.
        Under bounds the outer coordinates take the Coleman-Li coordinates of
        the outer part of the full gradient, as the joint mode's do. ``outer``
        and ``zero`` are the outer subproblem's coordinates and its solve at
        zero damping, already made: they are used when the coordinates agree.
        Returns ``(number, solves, step)``, ``step`` the reduced model's full
        step (``_FullStep``), which the gradient floor reads.
        """
        prefix = self.prefix
        bounds = self.route.bounds
        outer_gradient, _ = elimination.level_vectors(reduced)
        if bounds is None:
            coordinates = (
                _subproblem(point[:prefix], outer_gradient, scale[:prefix], None)
                if outer is None
                else outer
            )
        else:
            gradient, _ = elimination.level_vectors(system)
            coordinates = _subproblem(
                point[:prefix],
                gradient[:prefix],
                scale[:prefix],
                (bounds[0][:prefix], bounds[1][:prefix]),
            )
        solves = 0
        agree = (
            outer is not None
            and zero is not None
            and np.array_equal(coordinates.scale, outer.scale)
            and (
                (coordinates.addition is None and outer.addition is None)
                or (
                    coordinates.addition is not None
                    and outer.addition is not None
                    and np.array_equal(coordinates.addition, outer.addition)
                )
            )
        )
        if not agree:
            zero = elimination.solve_scaled(reduced, coordinates.scale, 0.0, coordinates.addition)
            solves = 1
        step = _full_step(reduced, outer_gradient, coordinates.scale, zero, coordinates.addition)
        total = step.decrease + float(np.sum(np.asarray(decreases, dtype=np.float64)))
        return total / relative_to, solves, step

    def member_quadratic(self, system: Any, coordinates: Any, member: int) -> Any:
        """``quadratic_form`` of one member's scaled subtree model, its diagonal addition in it.

        It takes a vector over the member's columns in the coordinates
        ``z = coordinates.scale * s`` and gives ``z^T (H_z + diag(addition)) z``.
        """
        columns = np.flatnonzero(self.owners == member)
        with np.errstate(all="ignore"):
            d = 1.0 / coordinates.scale
        addition = coordinates.addition[columns]

        def quadratic_form(vector: np.ndarray) -> float:
            full = np.zeros_like(d)
            with np.errstate(all="ignore"):
                full[columns] = d[columns] * vector
                quadratic = elimination.inner_quadratic(system, full, self.cut)[member]
                return float(quadratic) + float(addition @ (vector * vector))

        return quadratic_form

    def choose_members(
        self,
        system: Any,
        x: np.ndarray,
        gradient: np.ndarray,
        coordinates: Any,
        step: np.ndarray,
        radii: np.ndarray,
        active: np.ndarray,
        predicted: np.ndarray,
    ) -> tuple:
        """Every active member's trial step under the box, from its subproblem's solution.

        A member whose ``x + step`` stays in the box keeps its step and its
        predicted decrease; every other one takes ``_bounds.select_step`` on its
        scaled subtree model with its radius. Returns ``(step, scaled_step,
        predicted)`` in joint columns and per member.
        """
        lower, upper = self.route.bounds
        with np.errstate(all="ignore"):
            d = 1.0 / coordinates.scale
            scaled = coordinates.scale * step
        g_h = d * gradient
        chosen, chosen_scaled = step.copy(), scaled.copy()
        predicted = np.array(predicted, dtype=np.float64, copy=True)
        for member in np.flatnonzero(active):
            columns = np.flatnonzero(self.owners == member)
            if box.in_bounds(x[columns] + step[columns], lower[columns], upper[columns]):
                continue
            reach = max(float(radii[member]), float(np.linalg.norm(scaled[columns])))
            member_step, member_scaled, value = box.select_step(
                x[columns],
                self.member_quadratic(system, coordinates, int(member)),
                g_h[columns],
                step[columns],
                scaled[columns],
                d[columns],
                reach,
                lower[columns],
                upper[columns],
                box.step_back_ratio(gradient[columns], coordinates.v[columns]),
            )
            chosen[columns] = member_step
            chosen_scaled[columns] = member_scaled
            predicted[member] = float(value)
        return chosen, chosen_scaled, predicted

    def respond(
        self,
        system: Any,
        x: np.ndarray,
        scale: np.ndarray,
        multiplier: float,
        outer_step: Any,
        radii: np.ndarray,
    ) -> tuple:
        """The trial of an outer step: the inner coordinates moved by their response to it.

        The response is the inner part of the minimizer of the damped model
        with the outer step held, at the outer step's multiplier, bounded per
        member by its radius in ``radii``: a member whose response's scaled
        length exceeds its radius takes its own trust-region step of that
        radius on its subtree's model at the gradient the response answers
        (``_level_elimination.inner_bounded_response``). Under bounds it is
        solved in the Coleman-Li coordinates of the inner coordinates at
        ``x``, with their diagonal addition; every member whose bounded
        response leaves the box takes ``_bounds.select_step``'s choice on its
        scaled subtree model with the outer step held (its gradient the inner
        gradient that the response answers, its reach the bounded response's
        scaled length), and the trial is kept strictly inside the box.

        Returns ``(trial, norms, capped, solves)``: the trial point, every
        member's response norm before the bound, whether it was bounded, and
        the damped solves the bound made.
        """
        bounds = self.route.bounds
        if bounds is None:
            step, _, _, norms, capped, solves = elimination.inner_bounded_response(
                system, scale, multiplier, outer_step, radii
            )
            return x + step, norms, capped, solves
        lower, upper = bounds
        prefix = self.prefix
        gradient, _ = elimination.level_vectors(system)
        coordinates = box.coleman_li_scaling(x, gradient, lower, upper, scale)
        inner_scale = np.array(coordinates.scale, copy=True)
        inner_scale[:prefix] = scale[:prefix]
        addition = np.array(coordinates.addition, copy=True)
        addition[:prefix] = 0.0
        coordinates = coordinates._replace(scale=inner_scale, addition=addition)
        response, z, shifted, norms, capped, solves = elimination.inner_bounded_response(
            system, inner_scale, multiplier, outer_step, radii, addition=addition
        )
        with np.errstate(all="ignore"):
            d = 1.0 / inner_scale
        trial = x + response
        for member in range(len(self.members)):
            columns = np.flatnonzero(self.owners == member)
            if box.in_bounds(trial[columns], lower[columns], upper[columns]):
                continue
            member_step, _, _ = box.select_step(
                x[columns],
                self.member_quadratic(system, coordinates, member),
                shifted[columns],
                response[columns],
                z[columns],
                d[columns],
                float(np.linalg.norm(z[columns])),
                lower[columns],
                upper[columns],
                box.step_back_ratio(gradient[columns], coordinates.v[columns]),
            )
            trial[columns] = x[columns] + member_step
        return box.make_strictly_feasible(trial, lower, upper, rstep=0), norms, capped, solves

    def reduce(self, system: Any) -> Any:
        """The outer system: the inner blocks eliminated; a singular inner problem is refused.

        Under bounds the inner blocks carry the diagonal ``|g_i| / v_i`` of the
        Coleman-Li model wherever the gradient points at a finite side, so the
        outer model sees an inner coordinate held at its side as held.
        """
        bounds = self.route.bounds
        source = system
        if bounds is not None:
            gradient, _ = elimination.level_vectors(system)
            point = np.asarray(system.point)
            v, dv = box.CL_scaling_vector(point, gradient, bounds[0], bounds[1])
            inner = np.arange(point.shape[0]) >= self.prefix
            with np.errstate(all="ignore"):
                term = np.where(inner & (dv != 0), gradient * dv / v, 0.0)
            source = elimination.scaled_system(system, np.ones_like(point), term)
        reduced = elimination.reduce_to(source, self.cut)
        if _finite_blocks(reduced):
            return reduced
        singular = np.flatnonzero(elimination.inner_singular(system, self.cut))
        level = self.solver.nested_at
        if singular.size:
            member = self.members[int(singular[0])]
            raise ValueError(
                "the nested mode needs the Gauss-Newton matrix of every inner problem to be "
                f"nonsingular, and member {member!r} of level {level!r} has a singular one at "
                "this point; use nested_at=None."
            )
        raise ValueError(
            f"the Gauss-Newton matrix of the levels above {level!r}, with the inner problems "
            "eliminated, is not finite at this point; use nested_at=None."
        )

    def blocks(self, point: np.ndarray, *, start: bool) -> tuple:
        """The joint fit's ``blocks``; the first one also sets ``owners``, once.

        ``owners`` is the member of the cut level whose subtree holds each
        joint column (``-1`` above it). The fit's first request is this one,
        at the start or the restored point, before every method that reads it.
        """
        system, lane_costs = super().blocks(point, start=start)
        if self.owners is None:
            self.owners = elimination.node_of_columns(system, self.cut)
        return system, lane_costs

    def nonfinite_block_failures(self, results: Any, system: Any) -> tuple:
        """The lanes of every member whose Gauss-Newton blocks or gradient are not finite at a finite cost.

        Each of the member's lanes is a failure of kind ``NONFINITE_BLOCKS``: its solve reached
        the end at a finite cost and its sensitivities are not finite there.
        """
        broken = elimination.nonfinite_members(system, self.cut)
        if not broken.any():
            return ()
        route = self.route
        costs = np.asarray(results.cost, dtype=np.float64)[0]
        records = []
        for member in np.flatnonzero(broken):
            positions = np.flatnonzero(self.lane_member == member)
            if not np.all(np.isfinite(costs[positions])):
                continue
            for position in positions:
                row = route.nodes[position]
                nodes = tuple(
                    str(route.members[level][int(row[level])]) for level in range(route.depth)
                )
                rungs = (int(self.work.lane_rungs[position]),)
                records.append(
                    _Failure(str(route.ids[position]), NONFINITE_BLOCKS, float("nan"), nodes, rungs)
                )
        return tuple(records)

    def failed_members(self, failures: tuple) -> np.ndarray:
        """The members of the cut level that the failed lanes belong to, each once, in order."""
        position = self.route.position
        return np.unique(
            np.asarray(
                [self.lane_member[position[failure.lane]] for failure in failures], dtype=np.int64
            )
        )

    def retreating_blocks(
        self,
        point: np.ndarray,
        anchor: np.ndarray,
        radii: np.ndarray,
        retreats: np.ndarray,
        attempt: int,
    ) -> tuple:
        """The ``blocks`` request at an inner point of an outer trial, the members retreating.

        ``point`` is the responded point or a point the polish reached, and
        ``anchor`` holds every member's coordinates at its last point where
        its lanes solved: the current point for the responded point, the
        point before the polish's last iteration otherwise. A member with a
        lane that no rung solves at ``point`` retreats along its move from
        ``anchor``: at its ``j``-th retreat its subtree's coordinates are
        ``anchor + (point - anchor) / 2**j``, its radius is quartered, and
        only its lanes are solved again there, in one ``value`` request at
        the rungs they hold with every other coordinate as it was, up to the
        solver's ``inner_retreats`` times. When the lanes of every member that
        retreated solve, the ``blocks`` request is made again at the
        retreated point; a member whose lane fails there again goes on with
        the retreats it has left. A retreated point that the trial predicate
        rejects counts as a retreat at which the member's lanes failed.
        ``radii`` and ``retreats`` (the retreats of the attempt, per member)
        are updated in place, and every failure a retreat answers is logged
        as the member's. A failed lane of weight zero is logged as ignored and
        makes no member retreat.

        Returns ``(point, results, system, failures)``: the point where every
        lane solved, the results and system there, and ``()``; or, when a
        member cannot be solved at any retreat, the last point tried,
        ``None``, ``None`` and the failures of the last request.
        """
        route, work = self.route, self.work
        allowed = int(self.solver.inner_retreats)
        owners = self.owners
        below = owners >= 0
        member_of = np.maximum(owners, 0)
        anchor = np.asarray(anchor)
        failing = np.array(point, copy=True)
        point = np.array(point, copy=True)
        tried = np.zeros((len(self.members),), dtype=np.int64)
        while True:
            results = self.evaluate("blocks", point)
            failures, ignored = route.weigh(route.failures(results))
            work.record(attempt, ignored, inner=True)
            if not failures:
                system = route.system(results, point)
                # A member whose lanes solved at a finite cost but whose Gauss-Newton blocks
                # are not finite fails like a member whose lanes did not solve.
                failures, ignored = route.weigh(self.nonfinite_block_failures(results, system))
                work.record(attempt, ignored, inner=True)
            if not failures:
                work.held = system.held
                # The lanes' terms of the gradient floor's bound at this point.
                self.floor.record(results)
                return point, results, system, ()
            pending = self.failed_members(failures)
            unlogged = failures
            while pending.size:
                if np.any(tried[pending] >= allowed):
                    return point, None, None, failures
                if unlogged:
                    work.record(attempt, unlogged, inner=True)
                    unlogged = ()
                tried[pending] += 1
                retreats[pending] += 1
                work.inner_retreats[pending] += 1
                radii[pending] = np.maximum(radii[pending] / 4.0, _TINY)
                moving = below & np.isin(member_of, pending)
                halving = 0.5 ** tried[member_of[moving]]
                point[moving] = anchor[moving] + (failing[moving] - anchor[moving]) * halving
                lanes = tuple(
                    route.ids[position]
                    for position in np.flatnonzero(np.isin(self.lane_member, pending))
                )
                if not self.valid(point):
                    work.predicate_rejections += 1
                    continue
                retried = self.evaluate("value", point, lanes)
                failures, ignored = route.weigh(route.failures(retried))
                work.record(attempt, ignored, inner=True)
                unlogged = failures
                pending = self.failed_members(failures) if failures else pending[:0]

    def inner_solve(
        self,
        point: np.ndarray,
        *,
        origin: Any = None,
        scale: Any,
        radii: np.ndarray,
        locks: np.ndarray,
        thresholds: np.ndarray,
        attempt: int,
        start: bool,
        limit: int,
        outer_predicted: Any = None,
        noise: float = 0.0,
        level: Any = None,
    ) -> _InnerResult:
        """Every member of the cut level iterates on its subtree at the outer point of ``point``.

        The members start from their coordinates in ``point`` with the radii
        ``radii`` and the growth locks ``locks`` they carry, and make at most
        ``limit`` lockstep iterations; ``limit=0`` evaluates ``point`` alone.
        ``start`` marks the start point of a fresh fit, where a lane that fails
        at any ``blocks`` request is an error. At an outer trial a member whose
        lane fails at the first request retreats along its response toward its
        coordinates in ``origin`` (the current point; ``None`` takes
        ``point``), and one whose lane fails at the ``blocks`` request after
        an iteration retreats along its last step (``retreating_blocks``); a
        member that no retreat solves ends the solve with its failures. A
        member is done when its
        stationarity number is at most the inner ``stationarity_tol``, when its
        gradient is within the inner ``grad_tol`` (when that is set), when the
        decrease its model predicts for its full step is at most
        ``inner_fraction`` times ``outer_predicted`` (the decrease the outer
        model predicted for the outer step; ``None`` tests nothing), by its
        step and cost tests, when three of its rejected trials in a row within
        these iterations are dominated by the noise of the cost's evaluation,
        or when its radius falls to the smallest positive float. The
        tolerance's default reads ``noise``, the evaluation noise the outer
        trials measured, and so do the members' gain ratios under the inner
        ``noise_relaxed_ratio``. With ``level``, the gradient floor the outer
        trials measured (``None`` before the first sample), a member is also
        done when its stationarity number is at most its own floor, ``level``
        times its full step's scaled norm over its cost scale
        ``max(|cost_n|, cost_atol / cost_tol)`` under the inner tolerances.
        """
        solver, inner, route, work = self.solver, self.inner, self.route, self.work
        tolerances = self.inner_tolerances
        count = len(self.members)
        x = np.array(point, copy=True)
        scale = None if scale is None else np.array(scale, copy=True)
        radii = np.array(radii, dtype=np.float64, copy=True)
        locks = np.array(locks, dtype=np.int64, copy=True)
        thresholds = np.array(thresholds, dtype=np.float64, copy=True)
        iterations = np.zeros((count,), dtype=np.int64)
        bounds = route.bounds
        fixed = np.asarray(route.fixed, dtype=np.int64)
        tiny = float(np.finfo(float).tiny)
        member_limit = _stationarity_limit(tolerances.stationarity_tol, tolerances.cost_tol, noise)
        noise_limit = _noise_trial_limit(inner, route)
        accuracy = (
            None if outer_predicted is None else solver.inner_fraction * float(outer_predicted)
        )

        # Every member's retreats in this solve.
        retreats = np.zeros((count,), dtype=np.int64)
        # The lane costs at the current inner point, with those of a consistency re-solve.
        if start:
            system, lane_costs = self.blocks(x, start=True)
        else:
            # The responded point: a member whose lane fails there retreats along its
            # response toward its coordinates at the current point.
            x, results, system, failures = self.retreating_blocks(
                x, x if origin is None else origin, radii, retreats, attempt
            )
            if failures:
                return _InnerResult(
                    x,
                    None,
                    scale,
                    radii,
                    locks,
                    thresholds,
                    iterations,
                    failures,
                    retreats=retreats,
                )
            lane_costs = np.array(results.cost, dtype=np.float64)[0]
        first_cost = float(system.cost)
        owners = self.owners
        below = owners >= 0
        member_of = np.maximum(owners, 0)
        held = np.array(system.held_mask[self.cut], dtype=bool)
        active = ~held
        floored = np.zeros((count,), dtype=bool)
        codes = np.full((count,), _NOT_POLISHED, dtype=np.int64)
        # Per member: its noise-dominated rejected trials in a row, and the radius of its
        # last accepted trial in these iterations (nan before the first).
        noise_runs = np.zeros((count,), dtype=np.int64)
        accepted_radii = np.full((count,), np.nan)

        def per_member(values: np.ndarray) -> np.ndarray:
            return np.sqrt(np.bincount(owners[below], weights=values[below] ** 2, minlength=count))

        def finish(done: np.ndarray, code: int) -> None:
            codes[done] = code
            active[done] = False

        k = 0
        while True:
            # (2) the scale; (3) every member's gradient, its threshold, who is done
            gradient, diagonal = self.route.vectors(system)
            candidate = _scale_candidate(system, solver.x_scale, diagonal)
            scale = candidate if scale is None else np.maximum(scale, candidate)
            costs = self.member_costs(lane_costs, x)
            # Every member's subproblem coordinates; under bounds shaped by the
            # distance to the sides its gradient points at.
            coordinates = _subproblem(x, gradient, scale, bounds)
            # A member's gradient test, when the inner grad_tol is set; without it the
            # members' stationarity numbers below are their test.
            if tolerances.grad_tolerance is not None:
                norms = per_member(_bounded_gradient(gradient, coordinates, bounds))
                for member in np.flatnonzero(np.isnan(thresholds) & ~held):
                    thresholds[member] = tolerances.grad_tolerance.threshold(norms[member])
                finish(active & (norms <= thresholds), _GRADIENT_TOLERANCE)
            # Every member's solve at zero damping: its predicted decrease and its
            # stationarity number at the current point, and the first solve of the
            # active members' searches.
            zero = elimination.inner_solve_scaled(
                system, coordinates.scale, self.cut, addition=coordinates.addition
            )
            decreases = self.member_decrease(system, coordinates.scale, zero, coordinates.addition)
            stationarity = _member_numbers(decreases, self.member_scales(costs))
            full_steps = self.member_full_steps(zero, costs)
            # A member whose model predicts no further decrease the evaluation can
            # show is done, as one whose gradient is within its threshold, and as
            # one that has less to gain than inner_fraction of the outer step; once
            # the gradient floor is measured, so is one whose model predicts no more
            # than its derivatives resolve.
            limits = member_limit if level is None else np.fmax(member_limit, level * full_steps)
            finish(active & (stationarity <= limits), _STATIONARITY)
            if accuracy is not None:
                finish(active & (decreases <= accuracy), _ACCURACY)
            # (4) the exit always follows a blocks request at the current point
            if not active.any() or k == int(limit):
                # the exit makes no step; its one solve is the one above
                work.solves += 1
                break
            # (5) every active member's step; (6) bounds, fixed coordinates, predicate
            step, predicted, multipliers, solves = elimination.inner_secular_step(
                system,
                coordinates.scale,
                radii,
                self.cut,
                active,
                first=zero,
                addition=coordinates.addition,
                hold=True,
            )
            work.solves += solves
            # A member with no finite step inside its radius (its subtree's model is not
            # finite) ends its polish at its current point; the others go on.
            finish(active & np.isnan(multipliers), _NO_FINITE_STEP)
            if not active.any():
                break
            if fixed.size:
                step[fixed] = 0.0
            moving = below & active[member_of]
            trial = x.copy()
            if bounds is None:
                trial[moving] = x[moving] + step[moving]
                scaled = coordinates.scale * step
            else:
                # Every member whose step leaves the box takes the best of its step,
                # the step reflected at the sides it crosses and the Cauchy step, by
                # its model's value; its trial is kept strictly inside the box.
                step, scaled, predicted = self.choose_members(
                    system, x, gradient, coordinates, step, radii, active, predicted
                )
                trial[moving] = box.make_strictly_feasible(
                    x[moving] + step[moving], bounds[0][moving], bounds[1][moving], rstep=0
                )
            step = trial - x
            step_norms = per_member(step)
            scaled_norms = per_member(scaled)
            passing = active.copy()
            for member in np.flatnonzero(active):
                single = x.copy()
                columns = owners == member
                single[columns] = trial[columns]
                if not self.valid(single):
                    passing[member] = False
                    work.predicate_rejections += 1
            combined = x.copy()
            chosen = below & passing[member_of]
            combined[chosen] = trial[chosen]
            if passing.any() and not self.valid(combined):
                work.predicate_rejections += int(passing.sum())
                passing[:] = False
            # (7) one value request at the combined trial; a failed lane rejects its member
            rho = np.full((count,), -np.inf)
            failed = np.zeros((count,), dtype=bool)
            trial_costs = np.full((count,), np.inf)
            evaluated = bool(passing.any())
            if evaluated:
                before = work.lane_rungs.copy()
                trial_results = self.evaluate("value", combined)
                # A failed lane of weight zero rejects no member's step: logged, ignored.
                trial_failures, ignored = route.weigh(route.failures(trial_results))
                work.record(attempt, ignored, inner=True)
                raised = np.flatnonzero(work.lane_rungs > before)
                if raised.size:
                    # A lane that moved to a higher rung at the inner trial is solved again
                    # at the current inner point at its new rung, so that its member's gain
                    # ratio compares costs of the same rung.
                    lane_costs, _ = self.resolve(x, raised, before, lane_costs)
                    costs = self.member_costs(lane_costs, x)
                for failure in trial_failures:
                    failed[self.lane_member[route.position[failure.lane]]] = True
                work.record(attempt, trial_failures, inner=True)
                trial_costs = self.member_costs(
                    np.asarray(trial_results.cost, dtype=np.float64)[0], combined
                )
                gains = passing & ~failed & (predicted > 0.0)
                # Every member's gain ratio; relaxed by the noise the outer trials
                # measured when the inner noise_relaxed_ratio is set.
                rho[gains] = _gain_ratio(
                    costs[gains] - trial_costs[gains],
                    predicted[gains],
                    noise=noise,
                    relative_to=self.member_scales(costs[gains]),
                    relaxed=inner.noise_relaxed_ratio,
                )
            # (8) every member's acceptance, radius and lock, as the joint attempt's
            accepted = active & (rho >= float(inner.eta))
            step_radii = radii.copy()
            for member in np.flatnonzero(active):
                radii[member] = _next_radius(
                    float(radii[member]),
                    float(rho[member]),
                    float(scaled_norms[member]),
                    lock=int(locks[member]),
                    radius_max=inner.radius_max,
                    computed=bool(evaluated and passing[member] and not failed[member]),
                )
                if accepted[member]:
                    if locks[member]:
                        locks[member] -= 1
                elif failed[member]:
                    locks[member] = 2
            iterations[active] += 1
            work.inner_iterations[active] += 1
            if start:
                work.inner_start_iterations[active] += 1
            work.inner_rejections[active & ~accepted] += 1
            taken = below & accepted[member_of]
            previous = x.copy()
            x[taken] = trial[taken]
            # A member's step and cost tests read its trial whenever the trial's cost
            # was computed, accepted or rejected: the step within the step tolerance,
            # or the cost change and the decrease its model predicted both within the
            # cost tolerance (for an accepted trial the cost change alone). A trial the
            # predicate rejected, or whose lanes did not all solve, tests neither. Its
            # run of noise-dominated rejected trials, as the outer loop's: a trial whose
            # cost was computed extends or ends it, a failed one ends it, one the
            # predicate rejected leaves it; three in a row end its iterations.
            for member in np.flatnonzero(active):
                computed = bool(
                    evaluated
                    and passing[member]
                    and not failed[member]
                    and np.isfinite(trial_costs[member])
                )
                last = accepted_radii[member]
                noisy = _noise_dominated(
                    evaluated=computed,
                    accepted=bool(accepted[member]),
                    step_radius=float(step_radii[member]),
                    accepted_radius=None if np.isnan(last) else float(last),
                    cost=float(costs[member]),
                    trial_cost=float(trial_costs[member]),
                    predicted=float(predicted[member]),
                    relative_to=float(self.member_scales(costs[member])),
                    limit=noise_limit,
                )
                if computed:
                    noise_runs[member] = noise_runs[member] + 1 if noisy else 0
                elif failed[member]:
                    noise_runs[member] = 0
                if accepted[member]:
                    accepted_radii[member] = step_radii[member]
                # The loop's own step and cost tests, at the member's point after the
                # trial and at the member's cost threshold.
                if step_small(
                    float(step_norms[member]),
                    x[owners == member],
                    tolerances.step_tol,
                    tested=computed,
                ):
                    codes[member] = _STEP_TOLERANCE
                    active[member] = False
                elif cost_small(
                    accepted=bool(accepted[member]),
                    evaluated=computed,
                    cost_change=float(abs(trial_costs[member] - costs[member])),
                    predicted=float(predicted[member]),
                    threshold=cost_threshold(
                        costs[member], tolerances.cost_tol, tolerances.cost_atol
                    ),
                ):
                    codes[member] = _COST_TOLERANCE
                    active[member] = False
                elif noise_runs[member] >= _NOISE_RUN:
                    codes[member] = _EVALUATION_NOISE
                    active[member] = False
            floor = active & (radii <= tiny)
            floored |= floor
            finish(floor, _RADIUS)
            k += 1
            if evaluated:
                # (1) of the next iteration: the members' new point, every lane solved
                # there. At an outer trial a member whose lane fails there retreats along
                # its last step; at the start point the failure is an error.
                if start:
                    system, lane_costs = self.blocks(x, start=False)
                else:
                    x, results, system, failures = self.retreating_blocks(
                        x, previous, radii, retreats, attempt
                    )
                    if failures:
                        return _InnerResult(
                            x,
                            None,
                            scale,
                            radii,
                            locks,
                            thresholds,
                            iterations,
                            failures,
                            retreats=retreats,
                        )
                    lane_costs = np.array(results.cost, dtype=np.float64)[0]
        if int(limit) == 0:
            codes[~held] = _SKIPPED
        else:
            codes[active] = _AT_MAX_STEPS
            work.inner_unconverged[active | floored] += 1
        return _InnerResult(
            x,
            system,
            scale,
            radii,
            locks,
            thresholds,
            iterations,
            (),
            stationarity,
            lane_costs,
            decreases,
            codes,
            first_cost,
            retreats,
            full_steps,
            self.floor.latest,
        )

    def run(self, state: Any = None) -> OptResult:
        """Fit from the start point, or continue the fit whose ``state`` is given."""
        solver, inner, route, work = self.solver, self.inner, self.route, self.work
        dtype = np.asarray(route.problem.x0).dtype
        x0 = np.array(route.problem.x0, dtype=dtype)
        bounds = route.bounds
        count = len(self.members)
        prefix = self.prefix
        # grad_tol=None tests no gradient, in the outer loop and in the inner fits it sets.
        tolerances = loop_tolerances(
            dtype,
            grad_tol=solver.grad_tol,
            step_tol=solver.step_tol,
            cost_tol=solver.cost_tol,
            cost_atol=solver.cost_atol,
            stationarity_tol=solver.stationarity_tol,
            label=_LABEL,
        )
        grad_tolerance, step_tol, cost_tol, cost_atol, _ = tolerances
        # The inner fits' tolerances, read by the polish and the members' tests: the
        # outer ones when no inner solver is given.
        self.inner_tolerances = (
            tolerances
            if solver.inner is None
            else loop_tolerances(
                dtype,
                grad_tol=inner.grad_tol,
                step_tol=inner.step_tol,
                cost_tol=inner.cost_tol,
                cost_atol=inner.cost_atol,
                stationarity_tol=inner.stationarity_tol,
                label=_LABEL,
            )
        )
        outer_fixed = np.asarray([index for index in route.fixed if index < prefix], dtype=np.int64)
        outer_bounds = None if bounds is None else (bounds[0][:prefix], bounds[1][:prefix])
        n_constraints = int(route.constraint_matrix.shape[0])
        route.problem.prepare(modes=("value", "blocks"))
        start_iterations = np.zeros((count,), dtype=np.int64)
        start_codes = np.full((count,), _NOT_POLISHED, dtype=np.int64)
        # The polish's switch: off after two accepted attempts in a row whose polish
        # gained too little, on again after an outer rejection or a failed lane.
        polish_off, small_gains = False, 0
        if state is None:
            x = x0.copy()
            if bounds is not None:
                # Iterates stay strictly inside the box: a start on or beyond a side
                # moves to the next float inside it.
                x = box.make_strictly_feasible(x, bounds[0], bounds[1], rstep=0)
            if not trial_is_valid(route.view, x):
                raise ValueError("TrustRegion initial point failed the trial predicate.")
            radii = np.full((count,), float(inner.radius_init), dtype=np.float64)
            locks = np.zeros((count,), dtype=np.int64)
            thresholds = np.full((count,), np.nan)
            if solver.inner_start_iterations > 0:
                # At most inner_start_iterations lockstep iterations at the start point,
                # within the members' radii, before the first outer attempt.
                solved = self.inner_solve(
                    x,
                    scale=None,
                    radii=radii,
                    locks=locks,
                    thresholds=thresholds,
                    attempt=-1,
                    start=True,
                    limit=solver.inner_start_iterations,
                )
                x, system, scale = solved.point, solved.system, solved.scale
                radii, locks, thresholds = solved.radii, solved.locks, solved.thresholds
                start_iterations, start_codes = solved.iterations, solved.codes
                member_numbers, member_decreases = solved.stationarity, solved.decreases
                member_steps = solved.full_steps
                lane_costs = solved.lane_costs
            else:
                # No inner fit at the start point: the first outer attempt reads the
                # full system there.
                system, lane_costs = self.blocks(x, start=True)
                scale = _scale_candidate(
                    system, solver.x_scale, elimination.level_vectors(system)[1]
                )
                member_decreases, member_numbers, member_steps = self.members_at(
                    system, scale, x, lane_costs
                )
                work.solves += 1
        else:
            # The saved point is the start or an accepted point, after its polish; its
            # blocks are evaluated again. The members' radii and locks, and the
            # polish's switch, are those the saved state carries.
            x = np.array(state["point"], dtype=dtype)
            work.restore(state)
            work.resume_blocks += 1
            system, lane_costs = self.blocks(x, start=False)
            scale = np.array(state["scale"], dtype=dtype)
            radii = np.array(state["inner_radius"], dtype=np.float64)
            locks = np.array(state["inner_lock"], dtype=np.int64)
            thresholds = np.array(state["inner_grad_threshold"], dtype=np.float64)
            # A state saved without the switch resumes with the polish on.
            polish_off = bool(_saved_count(state, "polish_off"))
            small_gains = _saved_count(state, "polish_small_gains")
        cost = float(system.cost)
        reduced = self.reduce(system)
        # Every program an attempt runs is compiled here, before the loop: the outer
        # solve, quadratic and vectors of the reduced system, the inner response to
        # an outer step, and the inner solves.
        elimination.level_vectors(reduced)
        elimination.solve_scaled(reduced, np.ones((prefix,), dtype=dtype), 0.0)
        elimination.level_quadratic(reduced, np.zeros((prefix,), dtype=dtype))
        elimination.inner_response(
            system, scale, 0.0, np.zeros((prefix,), dtype=dtype), addition=np.zeros_like(scale)
        )
        elimination.inner_programs(system, self.cut)
        if state is not None:
            # The members' predicted decreases and stationarity numbers at the
            # restored point, from the restored scale, as its polish ended there.
            member_decreases, member_numbers, member_steps = self.members_at(
                system, scale, x, lane_costs
            )

        for callback in self.callbacks:
            callback.on_start(route.n, route.view.n_residuals, solver.max_steps)
        self.started = time.perf_counter()
        if state is None:
            # A fresh fit's first record holds the retries of the start's requests.
            record = self.record(
                float("nan"),
                float("nan"),
                False,
                x,
                (),
                start_iterations,
                start_codes,
                **work.attempt_records(),
            )
            loop = LoopState.start(
                point=x, cost=cost, scale=scale, radius=float(solver.radius_init), record=record
            )
        else:
            # A resumed fit's first record holds the retries of its first attempt.
            loop = LoopState.restore(state, point=x, cost=cost, scale=scale)
        # The largest relative predicted decrease of a trial counted as noise-dominated;
        # None bounds none.
        noise_limit = _noise_trial_limit(solver, route)
        start_attempt = loop.attempt
        attempt = start_attempt
        for attempt in range(start_attempt, int(solver.max_steps)):
            # The outer gradient with the inner gradients folded in: the gradient of
            # the model the outer step minimizes, and to first order the gradient of
            # the cost after the inner fits.
            outer_gradient, _ = elimination.level_vectors(reduced)
            x_outer = loop.point[:prefix]
            grad_norm = float(np.linalg.norm(outer_gradient))
            # The outer subproblem's coordinates, in the outer part of the scale.
            outer = _subproblem(x_outer, outer_gradient, loop.scale[:prefix], outer_bounds)
            stop_gradient = float(
                np.linalg.norm(_bounded_gradient(outer_gradient, outer, outer_bounds))
            )
            loop.cost_history.append(loop.cost)
            loop.grad_history.append(grad_norm)
            if grad_tolerance is not None:
                if loop.grad_threshold is None:
                    loop.grad_threshold = grad_tolerance.threshold(stop_gradient)
                if not n_constraints and stop_gradient <= loop.grad_threshold:
                    stop = "gradient_tolerance"
                    break

            # The outer step on the reduced system, in the outer part of the scale. Its
            # solve at lambda = 0 is the first solve of the step's multiplier search;
            # with every member's decrease at zero damping it gives the stationarity
            # number of the full system at the current point.
            zero = elimination.solve_scaled(reduced, outer.scale, 0.0, outer.addition)
            # Every relative number of the attempt is relative to the cost scale here.
            relative_to = cost_scale(loop.cost, cost_tol, cost_atol)
            stationarity, solves, full_step = self.full_stationarity(
                system,
                reduced,
                loop.point,
                loop.scale,
                relative_to,
                member_decreases,
                outer=outer,
                zero=zero,
            )
            work.solves += solves
            # The gradient's accuracy floor here, as in the joint mode; the bound reads
            # every coordinate, the outer ones in the outer step's coordinates. Every
            # member's floor is the fit's floor along the member's own full step.
            bound = self.floor.bound(np.concatenate((full_step.scale, loop.scale[prefix:])))
            level = _floor_level(loop.gradient_floor_samples, bound)
            floors = None if level is None else level * member_steps
            # The full model predicts no further decrease the evaluation can show here,
            # and neither does any member's model, against the members' yardstick.
            if stationarity <= _stationarity_limit(
                solver.stationarity_tol, cost_tol, _median(loop.noise_dominated_cost_change)
            ) and self.members_stationary(
                member_numbers, _median(loop.noise_dominated_cost_change), floors
            ):
                work.solves += 1
                stop = "stationarity"
                break
            # The full model predicts no further decrease its derivatives resolve here,
            # and neither does any member's model, against its floor.
            if (
                level is not None
                and stationarity <= _floor_sigma(level, full_step.norm, relative_to)
                and self.member_above(
                    member_numbers, _median(loop.noise_dominated_cost_change), floors=floors
                )
                is None
            ):
                work.solves += 1
                stop = "derivative_accuracy"
                break
            radius_floor = _radius_floor(level, full_step.norm, outer_gradient, outer.scale)
            current_members = np.array(member_numbers, dtype=np.float64)
            step_radius = loop.radius
            trial_outer, scaled_step_norm, predicted, multiplier, solves = _trial_point(
                reduced,
                x_outer,
                outer_gradient,
                outer,
                loop.radius,
                first=zero,
                fixed=outer_fixed,
                bounds=outer_bounds,
                equality=n_constraints > 0,
            )
            work.solves += solves
            step = trial_outer - x_outer
            # The trial's inner coordinates are the current ones moved by their
            # response to the outer step: the inner part of the minimizer of the
            # damped model with the outer step held, at the outer step's multiplier,
            # each member's bounded by its carried radius. The polish at the trial
            # starts there.
            trial, response_norms, response_capped, solves = self.respond(
                system, loop.point, loop.scale, multiplier, step, radii
            )
            work.solves += solves
            trial[:prefix] = trial_outer
            predicted_value = float(predicted)
            # The step test reads the trial's whole move, the outer step and the
            # members' response together, as the joint mode's reads its whole step.
            step_norm = float(np.linalg.norm(trial - loop.point))

            failures: tuple = ()
            failed_owners = None
            rejection_reason = None
            invalid = False
            solved = None
            iterations = np.zeros((count,), dtype=np.int64)
            codes = np.full((count,), _NOT_POLISHED, dtype=np.int64)
            retreats = np.zeros((count,), dtype=np.int64)
            logged = len(work.log)
            trial_ok = self.valid(trial)
            # A trial move within the step tolerance ends the fit at the current point
            # before the trial is evaluated.
            step_stop = step_small(step_norm, loop.point, step_tol, tested=trial_ok)
            if not trial_ok:
                work.predicate_rejections += 1
                trial_cost = float("inf")
                rejection_reason = "the trial predicate rejected the point"
            elif step_stop:
                trial_cost = float("inf")
                rejection_reason = "the step is within the step tolerance"
            else:
                # The polish at the trial, from the responded inner coordinates, with
                # the members' carried radii and locks; switched off, the responded
                # point alone is evaluated.
                before = work.lane_rungs.copy()
                if polish_off:
                    work.polish_skipped += 1
                solved = self.inner_solve(
                    trial,
                    origin=loop.point,
                    scale=loop.scale,
                    radii=radii,
                    locks=locks,
                    thresholds=thresholds,
                    attempt=attempt,
                    start=False,
                    limit=0 if polish_off else self.polish_steps,
                    outer_predicted=predicted_value,
                    noise=_median(loop.noise_dominated_cost_change),
                    level=level,
                )
                raised = np.flatnonzero(work.lane_rungs > before)
                if raised.size:
                    # A lane that moved to a higher rung in the polish at the trial is
                    # solved again at the current point at its new rung, so that the outer
                    # gain ratio compares costs of the same rung; the outer model there
                    # stays the earlier rung's.
                    lane_costs, change = self.resolve(loop.point, raised, before, lane_costs)
                    loop.cost += change
                iterations = solved.iterations
                trial = solved.point
                retreats = solved.retreats
                if solved.codes is not None:
                    codes = solved.codes
                if solved.failures:
                    invalid = True
                    failures = solved.failures
                    work.invalid_trials += 1
                    work.record(attempt, failures, inner=False)
                    trial_cost = float("inf")
                    failed_owners = route.owners(failures)
                    unsolvable = []
                    for failure in failures:
                        member = self.members[self.lane_member[route.position[failure.lane]]]
                        if member not in unsolvable:
                            unsolvable.append(member)
                    rejection_reason = (
                        f"the inner problem of {solver.nested_at} member(s) "
                        f"{_listed(unsolvable)} has no point where its lanes solve at the trial"
                    )
                    if solver.inner_retreats:
                        rejection_reason += (
                            f", after {solver.inner_retreats} retreat(s) toward the last point "
                            "where its lanes solved"
                        )
                else:
                    trial_cost = float(solved.system.cost)

            # The cost split by member at the current point and at the trial: read
            # off the current system and the polish's result, nothing of the fit changes.
            owners = self.owner_split(system, loop.point, lane_costs, solved)
            # A step within the step tolerance leaves the radius as it is.
            rho, accepted, loop.radius, reason = _judge(
                cost=loop.cost,
                trial_cost=trial_cost,
                predicted=predicted_value,
                valid=trial_ok,
                invalid=invalid,
                scaled_norm=scaled_step_norm,
                radius=loop.radius,
                lock=loop.growth_lock,
                noise_window=loop.noise_dominated_cost_change,
                eta=float(solver.eta),
                radius_max=solver.radius_max,
                relaxed=solver.noise_relaxed_ratio,
                relative_to=cost_scale(loop.cost, cost_tol, cost_atol),
                hold=step_stop,
                radius_floor=radius_floor,
            )
            assert_decision(accepted)
            rejection_reason = rejection_reason or reason

            previous_cost = loop.cost
            if accepted:
                # The point, its system and the members' state become the trial's; a
                # rejected trial leaves the members' radii and locks as they were
                # before its polish.
                loop.point, system = solved.point, solved.system
                loop.scale, radii, locks, thresholds = (
                    solved.scale,
                    solved.radii,
                    solved.locks,
                    solved.thresholds,
                )
                loop.cost = float(system.cost)
                lane_costs = solved.lane_costs
                reduced = self.reduce(system)
                member_numbers, member_decreases = solved.stationarity, solved.decreases
                member_steps = solved.full_steps
                # The lanes' terms of the gradient floor's bound move with the point.
                self.floor.current = solved.floor_terms
                loop.accepted += 1
            # The polish's switch: an accepted attempt whose polish made an iteration and
            # gained less than _POLISH_GAIN of the outer step's predicted decrease (the
            # cost at the responded point minus the cost after the polish) counts toward
            # switching it off, one that gained more resets the count, and one whose
            # polish made no iteration does neither; an outer rejection or a lane that
            # no rung solves turns it on.
            if not accepted or len(work.log) > logged:
                polish_off, small_gains = False, 0
            elif not polish_off and int(np.max(solved.iterations, initial=0)) > 0:
                gain = float(solved.first_cost) - float(solved.system.cost)
                small_gains = small_gains + 1 if gain < _POLISH_GAIN * predicted_value else 0
                if small_gains >= _SMALL_GAINS:
                    polish_off, small_gains = True, 0
            work.attempted(accepted)
            loop.radius_history.append(loop.radius)
            # An outer trial is evaluated when its cost was computed after its polish (a
            # step within the step tolerance is not); the outer trials alone measure the
            # evaluation noise and make the runs, the polish's trials do not enter.
            after = _after_attempt(
                loop,
                accepted=accepted,
                valid=trial_ok,
                invalid=invalid,
                cost=previous_cost,
                trial_cost=trial_cost,
                predicted=predicted_value,
                step_radius=step_radius,
                cost_tol=cost_tol,
                cost_atol=cost_atol,
                noise_limit=noise_limit,
                scaled_norm=scaled_step_norm,
                floor_bound=bound,
            )
            loop = dataclasses.replace(loop, **after.trials._asdict())
            loop.record = self.record(
                rho,
                float(multiplier),
                accepted,
                trial,
                failures,
                iterations,
                codes,
                retreats=retreats,
                capped=response_capped,
                norms=response_norms,
                owners=owners,
                **work.attempt_records(),
            )
            loop.attempt = attempt + 1
            info = self.step_info(
                loop,
                attempt,
                grad_norm=grad_norm,
                accepted=accepted,
                delta_norm=step_norm,
                rho=rho,
                lam=float(multiplier),
                rejection_reason=rejection_reason,
                failed_owners=failed_owners,
                state=self.state(
                    loop,
                    inner_radius=radii,
                    inner_lock=locks,
                    inner_grad_threshold=thresholds,
                    polish_off=polish_off,
                    polish_small_gains=small_gains,
                ),
                stationarity=stationarity,
                member_stationarity=current_members,
            )
            stop = _stop_after(
                after,
                requested=self.notify(info),
                max_invalid=int(solver.max_consecutive_invalid_trials),
                # The step test read the current point before the trial was evaluated.
                step_small=step_stop,
                accepted_count=loop.accepted,
                target_steps=solver.target_accepted_steps,
                # The budget counts the consistency re-solves and the inner fits' too.
                evaluations=work.blocks + work.value,
                max_evaluations=solver.max_evaluations,
            )
            if stop is not None:
                break
        else:
            attempt = max(int(solver.max_steps), start_attempt) - 1
            stop = "max_steps"

        gradient, _ = elimination.level_vectors(system)
        grad_norm = float(np.linalg.norm(gradient))
        projected_grad_norm = float(
            np.linalg.norm(
                _bounded_gradient(
                    gradient, _subproblem(loop.point, gradient, loop.scale, bounds), bounds
                )
            )
        )
        # The full system's stationarity number at the end point, from the members'
        # decreases there (those of the polish that ended there) and the outer part.
        end_relative_to = cost_scale(loop.cost, cost_tol, cost_atol)
        end_stationarity, _, end_step = self.full_stationarity(
            system, reduced, loop.point, loop.scale, end_relative_to, member_decreases
        )
        end_floor = self.floor_end(
            loop,
            end_step,
            np.concatenate((end_step.scale, loop.scale[prefix:])),
            cost_tol,
            end_relative_to,
            member_steps,
            system=system,
        )
        return self.outcome(
            loop,
            stop,
            attempt=attempt,
            x0=x0,
            cost_tol=cost_tol,
            grad_norm=grad_norm,
            projected_grad_norm=projected_grad_norm,
            stationarity=end_stationarity,
            members=member_numbers,
            floor=end_floor,
            inner_radius=radii,
            inner_lock=locks,
            inner_grad_threshold=thresholds,
            polish_off=polish_off,
            polish_small_gains=small_gains,
        )


__all__ = ["TrustRegion"]
