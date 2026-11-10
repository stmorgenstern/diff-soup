"""The lanes of a lowered study, the level tables that route them, and their checks."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Iterable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from diff_soup._errors import ConfigurationError
from diff_soup.optimization.inference.engine._layout import lane_layout
from diff_soup.optimization.inference.engine._levels import (
    LevelBlocks,
    LevelGradients,
    LevelTables,
    _lane_weights,
    member_row,
    reduce_level_gradients,
    reduce_levels,
    weigh_levels,
)
from diff_soup.optimization.inference.engine._plan import Plan
from diff_soup.optimization.inference.engine._reducers import refuse_failed
from diff_soup.optimization.inference.engine._request import Request
from diff_soup.optimization.inference.engine._status import LaneStatus
from diff_soup.optimization.inference.fitting.bounds import resolve_decision_bounds
from diff_soup.optimization.inference.hierarchical import _level_elimination as elimination
from diff_soup.optimization.inference.hierarchical._bounds import Scaling
from diff_soup.optimization.inference.hierarchical._constraints import _constraint_rows
from diff_soup.optimization.inference.hierarchical._level_system import regularizer_block_members

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


def _quoted(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in names)


def _listed(values: Iterable, limit: int = 8) -> str:
    values = tuple(values)
    text = ", ".join(str(value) for value in values[:limit])
    if len(values) > limit:
        text += f" and {len(values) - limit} more"
    return text


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


# ---------------------------------------------------------------------------
# The lanes of a lowered study
# ---------------------------------------------------------------------------


def _level_weights(weights: Any) -> tuple | None:
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


def _lane_positions(plan: Plan, ids: tuple) -> np.ndarray:
    """Per lane of ``ids``, its index in the study's lane layout, as the plan's buckets give it."""
    index = {
        lane: child for bucket in plan.buckets for lane, child in zip(bucket.lanes, bucket.children)
    }
    return np.asarray([index[lane] for lane in ids], dtype=np.int64)


def _experiment_level(problem: Any, ids: tuple, plan: Plan) -> tuple:
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


def _experiment_level_tables(
    problem: Any, ids: tuple, plan: Plan, device: jax.Device | None
) -> LevelTables:
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


def _owner_level_tables(
    problem: Any, view: Any, ids: tuple, plan: Plan, device: jax.Device | None
) -> LevelTables:
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


def _route_tables(
    problem: Any, view: Any, ids: tuple, plan: Plan, device: jax.Device | None
) -> tuple:
    """``(tables, members)``: the level tables of the lanes ``ids`` and the labels of every level.

    A [Hierarchy][diff_soup.optimization.inference.Hierarchy] study takes the tables of its
    hierarchy; a ``ParameterSet`` study with an owner tier the owners and the
    experiments; any other study one level of experiments. ``ids`` may be any
    subset of the planned lanes, in canonical order: the tables keep every
    node of the study, so the sums over disjoint subsets add up.
    """
    from diff_soup.optimization.inference.statistics.hierarchy import Hierarchy

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


def _member_row(results: Any, member: int) -> int:
    """The row of population member ``member`` in ``results``."""
    rows = np.flatnonzero(np.asarray(results.members) == int(member))
    if rows.shape[0] != 1:
        raise ValueError(
            f"member {int(member)} is not among the members of these results "
            f"{np.asarray(results.members).tolist()}; ask for one of them."
        )
    return int(rows[0])


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

    block_weights: tuple[float, ...] | None = None
    device: jax.Device | None = None
    # The gradient and diagonal of the last system, and the regularizer's blocks at the
    # last point it was evaluated at, each kept for the next request of the same.
    kept_vectors: tuple[elimination.LevelSystem, np.ndarray, np.ndarray] | None = None
    kept_regularizer: tuple[tuple[str, bytes, tuple[float, ...] | None], tuple] | None = None
    # The index tables of every level system of the fit, built and placed once.
    layout: elimination.LevelLayout | None = None
    # The level tables of subsets of the planned lanes, by their ids in canonical order.
    subset_tables: dict | None = None
    # The jitted programs of the regularizer's blocks: the route and the block weights they
    # were built for, and the programs by name (``programs``).
    compiled: tuple | None = None

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
                    self.attributions[number] = regularizer_block_members(
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
        [weigh_by][diff_soup.optimization.inference.hierarchical._lane_route._LaneRoute.weigh_by],
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
        lanes: Iterable[str] | None = None,
        rungs: ArrayLike | None = None,
        first_steps: ArrayLike | None = None,
    ) -> Any:
        """One request at one point: every planned lane, or the planned lanes ``lanes``.

        ``rungs`` and ``first_steps`` are the request's (one row, one column per
        requested lane): the rung each lane runs at and the first step of its
        rung-0 solve; ``None`` runs every lane at rung 0. A ``frozen`` request
        steps each lane on the Jacobian its last ``blocks`` request kept, once
        the study has prepared that mode
        ([gradients][diff_soup.optimization.inference.hierarchical._lane_route._LaneRoute.gradients],
        [frozen_system][diff_soup.optimization.inference.hierarchical._lane_route._LaneRoute.frozen_system]).
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

    def system(
        self, results: Any, point: np.ndarray, *, member: int = 0
    ) -> elimination.LevelSystem:
        """The level system at ``point`` from a ``blocks`` (or ``hessian``) evaluation there.

        ``member`` is the population member of ``results`` whose lanes are
        reduced; a fit's own requests hold the one member 0. Results of lanes
        evaluated in several processes carry the sums of the levels already
        (``Results.levels``, every lane weighing one); the route's weights are
        then applied to those sums. Results of a subset of the planned lanes
        give the system of those lanes alone (``subset_blocks``).
        """
        if tuple(results.lane_ids) != self.ids:
            blocks = self.subset_blocks(results, member)
        elif getattr(results, "levels", None) is not None:
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

    def tables_of(self, ids: tuple) -> Any:
        """The level tables of the planned lanes ``ids``, in canonical order; built once per subset.

        The tables keep every node of the study, so the sums of a subset fit
        the route's levels.
        """
        ids = tuple(ids)
        if ids == self.ids:
            return self.tables
        if self.subset_tables is None:
            self.subset_tables = {}
        tables = self.subset_tables.get(ids)
        if tables is None:
            tables, _ = _route_tables(
                self.problem, self.view, ids, self.problem.plan(), self.device
            )
            self.subset_tables[ids] = tables
        return tables

    def gradients(self, results: Any, *, member: int = 0) -> LevelGradients:
        """The weighted level gradients of the lanes of a ``gradient`` or ``frozen`` evaluation.

        ``results`` may hold any of the planned lanes; a node with none of
        them below it sums to zero. The lanes are weighed by the route's
        weights, and a failed lane or a unit left out of nonzero weight is
        refused, as ``system`` refuses them. Results of lanes evaluated in
        several processes carry the sums of the levels already
        (``Results.levels``); the weights are then applied to those sums.

        Returns:
            The [LevelGradients][diff_soup.optimization.inference.engine.LevelGradients]
            of ``member``, on the route's device.
        """
        ids = tuple(results.lane_ids)
        if getattr(results, "levels", None) is not None:
            row = _member_row(results, member)
            status = LaneStatus(
                kind=np.asarray(results.status.kind)[row],
                t_reached=np.asarray(results.status.t_reached)[row],
                rung=np.asarray(results.status.rung)[row],
            )
            omega = None
            if self.weights is not None:
                omega = self.omega[[self.position[lane] for lane in ids]]
            refuse_failed(status, ids, omega)
            sums = weigh_levels(self.tables, member_row(results.levels, row), self.weights)
            sums = LevelGradients(
                gradient=sums.gradient,
                gradient_shared=sums.gradient_shared,
                cost=sums.cost,
                rows=sums.rows,
            )
            return jax.device_put(sums, self.device)
        lanes = (
            results.lane_result(member)
            if self.device is None
            else results.lane_result(member, self.device)
        )
        return reduce_level_gradients(self.tables_of(ids), lanes, ids, self.weights)

    def frozen_system(
        self,
        system: elimination.LevelSystem,
        results: Any,
        point: np.ndarray,
        *,
        member: int = 0,
    ) -> elimination.LevelSystem:
        """``system`` with its matrix kept and its gradient at ``point`` from a ``frozen`` evaluation.

        Every node of a level whose lanes are all in ``results`` takes the
        weighted ``J^T r`` of its lanes
        ([gradients][diff_soup.optimization.inference.hierarchical._lane_route._LaneRoute.gradients])
        and the regularizer's slope at ``point``, zero at a fixed coordinate
        and at a held node of ``system``; so does the root when ``results``
        hold every planned lane. Every other node keeps the gradient of
        ``system``. The cost is the weighted lane costs and the regularizer's
        at ``point`` when ``results`` hold every planned lane, and NaN
        otherwise. At the point of ``system``, from a ``frozen`` evaluation of
        every lane, the system is ``system`` again.
        """
        sums = self.gradients(results, member=member)
        dtype = np.dtype(system.root.dtype)
        point = np.asarray(point, dtype=dtype)
        regularizer = self.regularizer_at(point, dtype)
        extra = None if regularizer is None else regularizer[0]
        layout = self.levels()
        counts, widths, offsets = layout.counts, layout.widths, layout.offsets
        free = ~np.asarray(system.fixed_mask, dtype=bool)
        positions = np.asarray([self.position[lane] for lane in results.lane_ids], dtype=np.int64)
        gradient = [None]
        for level in range(1, self.depth + 1):
            count = int(counts[level - 1])
            column = self.nodes[:, level - 1]
            covered = np.bincount(column[positions], minlength=count) == np.bincount(
                column, minlength=count
            )
            new = np.array(sums.gradient[level], dtype=dtype)
            if extra is not None and extra[4][level] is not None:
                new = new + np.asarray(extra[4][level], dtype=dtype)
            own = free[offsets[level] : offsets[level + 1]].reshape(count, int(widths[level]))
            own = own & ~np.asarray(system.held_mask[level], dtype=bool)[:, None]
            kept = np.asarray(system.gradient[level], dtype=dtype)
            gradient.append(np.where(covered[:, None], new * own, kept))
        if positions.shape[0] == len(self.ids):
            root = np.sum(np.asarray(sums.gradient_shared, dtype=dtype), axis=0)
            if extra is not None:
                root = root + np.asarray(extra[1], dtype=dtype)
            root = root * free[: int(widths[0])]
            cost = float(np.sum(np.asarray(sums.cost[self.depth], dtype=np.float64)))
            cost += 0.0 if regularizer is None else float(regularizer[1])
        else:
            root = np.asarray(system.root_gradient, dtype=dtype)
            cost = float("nan")
        root, gradient, point, cost = jax.device_put(
            (root, tuple(gradient), point, np.asarray(cost, dtype=dtype)), layout.device
        )
        return dataclasses.replace(
            system, root_gradient=root, gradient=gradient, point=point, cost=cost
        )

    def levels(self) -> elimination.LevelLayout:
        """The [LevelLayout][diff_soup.optimization.inference.hierarchical._level_system.LevelLayout] of the tables, built once."""
        if self.layout is None:
            self.layout = elimination.level_layout(self.tables, self.device)
        return self.layout

    def regularizer_at(self, point: np.ndarray, dtype: Any = None) -> tuple | None:
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

    def vectors(self, system: elimination.LevelSystem) -> tuple:
        """``elimination.level_vectors(system)``, kept for the last system asked for.

        The attempts of a fit that reject their trials step from the same
        system; its gradient and diagonal move to the host once. Copies are
        returned, so a caller may change them.
        """
        kept = self.kept_vectors
        if kept is None or kept[0] is not system:
            kept = self.kept_vectors = (system, *elimination.level_vectors(system))
        return kept[1].copy(), kept[2].copy()

    def solve_attempt(
        self, system: elimination.LevelSystem, coordinates: Scaling, radius: float
    ) -> elimination.AttemptSolve:
        """The solve at zero damping, its predicted decrease and the step of ``radius``, at once.

        ``elimination.attempt_solve`` in the subproblem's ``coordinates``: one
        program on the device and one transfer, in place of the chain of
        ``solve_scaled``, the full step's decrease and ``level_secular_step``.
        """
        return elimination.attempt_solve(system, coordinates.scale, radius, coordinates.addition)

    def _summed_levels(self, results: Any, member: int) -> LevelBlocks | LevelGradients:
        """The level blocks of ``member`` from results that carry the sums of the levels.

        The lanes' failures and unevaluated units are refused under the
        route's weights as ``reduce_levels`` refuses them; the sums, every
        lane weighing one, are then weighed by the route's weights.
        """
        row = _member_row(results, member)
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

    def programs(self) -> dict:
        """The route's jitted programs of the regularizer's blocks, by name.

        Built for this route and its block weights: a copy, or a route weighed
        again, builds its own.
        """
        kept = self.compiled
        if kept is None or kept[0] is not self or kept[1] != self.block_weights:
            kept = self.compiled = (self, self.block_weights, {})
        return kept[2]

    def refuse_block_count(self, count: int) -> None:
        """Refuse a regularizer that declares ``count`` blocks where its weights name another number."""
        weights = self.block_weights
        if weights is not None and count != len(weights):
            raise ValueError(
                f"the regularizer's block_rows(x) declares {count} blocks at this point "
                f"and {len(weights)} where its weights were attributed; a regularizer declares "
                "the same blocks at every point."
            )

    def block_costs(self, point: np.ndarray) -> tuple:
        """``(columns, costs)``: every regularizer block's columns and its ``0.5 |r_b|^2`` at ``point``.

        Unweighted, in the order of ``view.regularization.block_rows``, from one
        jitted program; ``costs`` is a host array.
        """
        programs = self.programs()
        if "costs" not in programs:
            regularizer = self.view.regularization
            columns: list = []

            def costs(x: jax.Array) -> jax.Array:
                declared = tuple(regularizer.block_rows(x))
                columns[:] = [tuple(int(column) for column in item[0]) for item in declared]
                rows = [jnp.ravel(item[1]) for item in declared]
                if not rows:
                    return jnp.zeros((0,), dtype=x.dtype)
                return jnp.stack([0.5 * (row @ row) for row in rows])

            programs["costs"] = (jax.jit(costs), columns)
        program, columns = programs["costs"]
        values = np.asarray(program(jax.device_put(point, self.device)))
        self.refuse_block_count(len(columns))
        return tuple(columns), values

    def penalty(self, point: np.ndarray) -> float:
        """The regularizer's cost at ``point``: every block's ``0.5 |r_b|^2`` times its weight."""
        regularizer = self.view.regularization
        if self.block_weights is None:
            return float(regularizer.penalty(jax.device_put(point, self.device)))
        _, costs = self.block_costs(point)
        weights = np.asarray(self.block_weights, dtype=np.float64)
        return float(np.sum(np.where(weights == 0.0, 0.0, weights * costs)))

    def penalty_gradient(self, point: np.ndarray) -> np.ndarray:
        """The gradient of ``penalty`` at ``point``: every block's ``J_b^T r_b`` times its weight.

        Each block of ``view.regularization.block_rows`` adds its slope to the
        joint columns it addresses, in one jitted program; a regularizer without
        rows gives zeros.
        """
        programs = self.programs()
        if "gradient" not in programs:
            regularizer, weights = self.view.regularization, self.block_weights

            def slope(x: jax.Array) -> jax.Array:
                gradient = jnp.zeros(x.shape, dtype=x.dtype)
                if int(np.size(regularizer.residual(x))) == 0:
                    return gradient
                declared = tuple(regularizer.block_rows(x))
                self.refuse_block_count(len(declared))
                for weight, (columns, residual, jacobian) in zip(
                    (1.0,) * len(declared) if weights is None else weights, declared
                ):
                    if weight == 0.0:
                        continue
                    residual = jnp.ravel(jnp.asarray(residual, dtype=x.dtype))
                    jacobian = jnp.asarray(jacobian, dtype=x.dtype).reshape(residual.size, -1)
                    at = np.asarray(columns, dtype=np.int64)
                    gradient = gradient.at[at].add(weight * (jacobian.T @ residual))
                return gradient

            programs["gradient"] = jax.jit(slope)
        point = np.asarray(point)
        return np.array(programs["gradient"](jax.device_put(point, self.device)), dtype=point.dtype)

    def trial_cost(self, results: Any, point: np.ndarray, *, member: int = 0) -> float:
        """The weighted lane costs of a ``value`` evaluation and the regularizer at ``point``.

        ``member`` is the population member of ``results`` whose lane costs
        are summed; a fit's own requests hold the one member 0.
        """
        costs = np.asarray(results.cost, dtype=np.float64)[_member_row(results, member)]
        lanes = float(np.sum(self.omega * costs))
        # The regularizer's cost from its blocks at the trial, which the system there reuses
        # when the trial is accepted: the regularizer is evaluated once per point.
        blocks = self.regularizer_at(point)
        return lanes + (0.0 if blocks is None else float(blocks[1]))

    def subset_blocks(self, results: Any, member: int = 0) -> LevelBlocks:
        """The weighted level blocks of a ``blocks`` evaluation of a subset of the planned lanes.

        A node with none of the lanes below it sums to zero. The lanes are
        weighed by the route's weights, and a failed lane or a unit left out
        of nonzero weight is refused, as ``system`` refuses them. Results of
        lanes evaluated in several processes carry the sums of the levels
        already (``Results.levels``); the weights are then applied to those
        sums.
        """
        ids = tuple(results.lane_ids)
        if getattr(results, "levels", None) is None:
            lanes = (
                results.lane_result(member)
                if self.device is None
                else results.lane_result(member, self.device)
            )
            return reduce_levels(self.tables_of(ids), lanes, ids, self.weights)
        row = _member_row(results, member)
        status = LaneStatus(
            kind=np.asarray(results.status.kind)[row],
            t_reached=np.asarray(results.status.t_reached)[row],
            rung=np.asarray(results.status.rung)[row],
        )
        omega = None
        if self.weights is not None:
            omega = self.omega[[self.position[lane] for lane in ids]]
        refuse_failed(status, ids, omega)
        sums = weigh_levels(self.tables, member_row(results.levels, row), self.weights)
        return jax.device_put(sums, self.device)

    def owners(self, failures: tuple) -> tuple:
        """The level-1 members of the failed lanes, each once, in lane order."""
        labels = []
        for failure in failures:
            label = self.members[0][int(self.nodes[self.position[failure.lane]][0])]
            if label not in labels:
                labels.append(label)
        return tuple(labels)
