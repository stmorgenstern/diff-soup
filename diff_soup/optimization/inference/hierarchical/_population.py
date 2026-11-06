"""The population route: one mode evaluated at many points at once, in lockstep.

A population is many points of one study evaluated together: population
member ``q`` is row ``q`` of every request. Every request sends the full
``(Q, n)`` array of points and masks the members that take no part, never
removes them, so the engine's step memory and its retry records name the same
member in every request. Every lane of every member holds a rung, started from
the rung the route was given (a fit's, for a route built at a fit) and raised,
never lowered, as a fit raises its own.

The requests go to the study cut for the population
(``LoweredStudy.for_population``): each bucket's program is as wide as the
population's units need, so the points run together, and every request's
``Results.batches`` records the units, the width and the calls of each bucket.
The route keeps those records, request by request, in ``requests``.

A route built at a fit (``PopulationRoute.at_fit``, or the constructor given
the fit) also weighs each member's objective on its own: the level system and
the trial cost of member ``q`` weigh its lanes, its regularizer blocks and its
gauge rows by its weights, through the same definition a weighted fit uses
(``_LaneRoute.weigh_by``). A bootstrap's replicates are such members; their
stepping is in ``_replicates``.

``Lockstep`` moves a population's members together: one attempt of every
member still running per iteration, each iteration's event recorded, given
to the checkpoint and to the callbacks. ``PopulationCheckpoint`` keeps a
population's iterations in one directory: the state of every member after an
iteration in one file, and one line per iteration in ``events.jsonl``. A
population restored from such a file forms its running members' systems with
one ``blocks`` request and continues.
"""

from __future__ import annotations

import json
import os
import re
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup.inference.engine._request import Request
from diff_soup.inference.hierarchical.level_solvers import TrustRegion, _hold_rungs, _LaneRoute

_MODES = ("value", "blocks")
_WORK = ("lane_ids", "lane_rungs", "lane_first_steps")


def _fit_binding(
    study: Any,
    fit: Any,
    population: int,
    *,
    over: Any,
    weight_priors: Any,
    chunk: Any,
    solver: Any,
) -> tuple:
    """What binds a route to a fit, checked: the cut study, the fit's lane route, the rungs.

    Returns ``(target, base, over, rungs, first_steps)``: the study cut for
    the population (``study`` itself when ``chunk`` is ``None``), the fit's
    lane route on it (whose fixed coordinates, constraints and gauges enter
    every member's system), the top level the weights weigh, and the rung and
    the rung-0 first step the fit holds for every lane, one row each in the
    order of the route's lanes.

    Raises:
        ConfigurationError: If ``weight_priors`` is not ``True`` or ``False``,
            ``over`` is not the top level, or ``solver`` is not a
            ``TrustRegion`` in the joint mode without weights.
        ValueError: If ``fit`` holds no rung for a lane of the study.
    """
    if weight_priors is not True and weight_priors is not False:
        raise ConfigurationError(f"weight_priors must be True or False; got {weight_priors!r}.")
    if solver is None:
        solver = TrustRegion()
    if not isinstance(solver, TrustRegion):
        raise ConfigurationError(
            f"solver= is a TrustRegion; got {type(solver).__name__}. Its fixed coordinates, "
            "constraints and gauges enter every member's system."
        )
    if solver.nested_at is not None:
        raise ConfigurationError(
            f"solver= has nested_at={solver.nested_at!r}; the population's systems are those "
            "of the joint mode, so give a TrustRegion with nested_at=None."
        )
    if solver.weights is not None:
        raise ConfigurationError(
            "solver= carries weights=; every population member is weighed by the weights "
            "given with it, so give a TrustRegion with weights=None."
        )
    work = getattr(fit, "work", None)
    if work is None or any(key not in work for key in _WORK):
        raise ValueError(
            "fit= is the result of TrustRegion on this lowered study, whose work holds every "
            "lane's rung and first step (fit.work['lane_rungs'], "
            "fit.work['lane_first_steps']); this fit carries none."
        )
    target = study if chunk is None else study.for_population(population, chunk=chunk)
    base = _LaneRoute(solver, target)
    top = base.names[0]
    if over is None:
        over = top
    elif isinstance(over, str):
        if over != top:
            raise ConfigurationError(
                f"over= names the level {over!r}; the population is weighed over the top "
                f"level of the tree, {top!r}."
            )
    elif isinstance(over, bool) or not isinstance(over, (int, np.integer)) or int(over) != 1:
        raise ConfigurationError(
            f"over= names the level {over!r}; the population is weighed over the top level "
            f"of the tree, {top!r} (level 1)."
        )
    held = [str(lane) for lane in np.ravel(work["lane_ids"])]
    missing = [lane for lane in base.ids if lane not in held]
    if missing:
        raise ValueError(
            f"the fit holds no rung for the lanes {missing[:8]} of this study; give the result "
            "of TrustRegion on this lowered study."
        )
    order = [held.index(lane) for lane in base.ids]
    rungs = np.asarray(work["lane_rungs"], dtype=np.int64)[order]
    first_steps = np.asarray(work["lane_first_steps"], dtype=np.float64)[order]
    return target, base, top, rungs, first_steps


class PopulationRoute:
    """Requests over a population of points; at a fit, each point's system and trial cost under its weights.

    Attributes:
        study: The study every request goes to: the study cut for the
            population, or the study itself when ``chunk=None``.
        population: ``Q``, the population members of every request.
        ids: The lanes, in the order of the study's experiments.
        position: Lane to its column in ``rungs`` and ``first_steps``.
        n: The coordinates of a point.
        rungs: ``(Q, L)``: per population member and lane, the rung it is
            asked for, lanes in the order of ``ids``.
        first_steps: ``(Q, L)``: the first step of the rung-0 solve each rung
            scales.
        requests: One read-only mapping per ``evaluate`` call, in order:
            ``mode``, ``members`` (the members that took part), ``lanes``
            (the lanes requested), ``batches`` (the request's
            ``Results.batches``) and ``seconds`` (its wall seconds).
        prepared: The report of the study's ``prepare()`` for the modes the
            route runs; ``None`` until a mode is prepared.

    A route built at a fit also holds, from the fit's lane route:

        over: The level whose members the weights weigh: the top level of
            the tree.
        weight_priors: Whether a member's regularizer blocks take its weight.
        tables: The level tables of the lanes.
        members: Per level, the labels of its members.
        names: The names of the levels.
        nodes: Per lane, its node at every level.
        bounds: The bounds of the fitted coordinates, or ``None``.
        fixed: The coordinates held fixed.
        view: The joint problem of the study.
        evaluation_noise: The ceiling of the noise of the cost's evaluation,
            or ``None`` when it is unknown.
    """

    def __init__(
        self,
        study: Any,
        fit: Any = None,
        *,
        population: int,
        lanes: Any = None,
        rungs: Any = None,
        first_steps: Any = None,
        chunk: Any = "auto",
        modes: tuple = _MODES,
        over: Any = None,
        weight_priors: bool = True,
        solver: Any = None,
    ):
        """Cut the study for the population, compile its programs, start every lane at its rung.

        Two forms. ``PopulationRoute(study, lanes=, population=, rungs=,
        first_steps=, chunk=, modes=)`` speaks of lanes, points, rungs and
        modes and binds the route to no fit: it makes requests (``evaluate``)
        and nothing else. ``PopulationRoute(study, fit, population=, over=,
        weight_priors=, chunk=, solver=, modes=)``, which ``at_fit`` builds by
        name, takes the lanes, rungs and first steps from the fit and weighs
        each member's level system and trial cost by its own weights
        (``system``, ``trial_cost``, ``failures``).

        Args:
            study: The lowered study (with ``fit``: the one the fit was made
                on).
            fit: ``None`` for the plain form; else the result of
                ``TrustRegion`` on ``study``, whose ``work`` gives every
                lane's rung and first step.
            population: ``Q``, the population members of every request, a
                positive integer.
            lanes: Plain form: the ids of the study's lanes, in the order of
                its experiments. Not given with ``fit``.
            rungs: Plain form: one rung per lane, in the order of ``lanes``;
                ``None`` starts every lane at rung 0. Not given with ``fit``.
            first_steps: Plain form: as ``rungs``, the first step of the
                rung-0 solve each rung scales; ``None`` holds 0 for every
                lane. Not given with ``fit``.
            chunk: The width of the population's programs, passed to
                ``study.for_population(population, chunk=)``: ``"auto"``,
                a positive integer, or ``None`` to send every request to
                ``study`` itself.
            modes: The modes whose programs ``prepare()`` compiles (or
                reuses) before the first request: ``("value", "blocks")``,
                ``("blocks",)`` for a route that evaluates blocks only, or
                ``()`` to prepare nothing yet (``prepare`` does it later).
            over: With ``fit``: the level whose members the weights weigh,
                by name or number; ``None`` or the top level of the tree,
                the only one accepted.
            weight_priors: With ``fit``: ``True`` weighs a member's
                regularizer blocks by its weight; ``False`` leaves every
                block unweighted.
            solver: With ``fit``: a ``TrustRegion`` in the joint mode without
                weights, whose fixed coordinates, constraints and gauges the
                systems take; ``None`` takes ``TrustRegion()``.

        Raises:
            ConfigurationError: If ``population`` is not a positive integer;
                in the plain form, if ``lanes`` is missing or ``over``,
                ``weight_priors`` or ``solver`` is given; with ``fit``, if
                ``lanes``, ``rungs`` or ``first_steps`` is given,
                ``weight_priors`` is not ``True`` or ``False``, ``over`` is
                not the top level, or ``solver`` is not a ``TrustRegion`` in
                the joint mode without weights.
            ValueError: In the plain form, if ``rungs`` or ``first_steps``
                has not one entry per lane; with ``fit``, if it holds no rung
                for a lane of the study.
        """
        if (
            isinstance(population, bool)
            or not isinstance(population, (int, np.integer))
            or population < 1
        ):
            raise ConfigurationError(
                f"population= is the number of points of every request, a positive integer; got "
                f"{population!r}."
            )
        population = int(population)
        if fit is None:
            if lanes is None:
                raise ConfigurationError(
                    "lanes= names the lanes of the study, in the order of its experiments; give "
                    "lanes=, or fit= to take them from a fit."
                )
            if over is not None or weight_priors is not True or solver is not None:
                raise ConfigurationError(
                    "over=, weight_priors= and solver= weigh a population at a fit; give them "
                    "with fit=."
                )
            target = study if chunk is None else study.for_population(population, chunk=chunk)
            ids = tuple(str(lane) for lane in lanes)
            n = int(study.n_params)
            count = len(ids)
            rungs = (
                np.zeros((count,), dtype=np.int64)
                if rungs is None
                else np.asarray(rungs, dtype=np.int64)
            )
            first_steps = (
                np.zeros((count,), dtype=np.float64)
                if first_steps is None
                else np.asarray(first_steps, dtype=np.float64)
            )
            if rungs.shape != (count,) or first_steps.shape != (count,):
                raise ValueError(
                    f"rungs= and first_steps= hold one entry per lane, shape ({count},); got "
                    f"{rungs.shape} and {first_steps.shape}."
                )
        else:
            if lanes is not None or rungs is not None or first_steps is not None:
                raise ConfigurationError(
                    "with fit=, the lanes, rungs and first steps are the fit's; give none of "
                    "lanes=, rungs= and first_steps=."
                )
            target, base, over, rungs, first_steps = _fit_binding(
                study,
                fit,
                population,
                over=over,
                weight_priors=weight_priors,
                chunk=chunk,
                solver=solver,
            )
            ids, n = base.ids, base.n
        self.study = target
        self.population = population
        self.ids = ids
        self.position = {lane: index for index, lane in enumerate(ids)}
        self.n = n
        self.rungs = np.tile(rungs, (population, 1))
        self.first_steps = np.tile(first_steps, (population, 1))
        self.requests: list = []
        self._cut = chunk is not None
        self.prepared = None
        if fit is not None:
            self.over = over
            self.weight_priors = bool(weight_priors)
            self.tables = base.tables
            self.members = base.members
            self.names = base.names
            self.nodes = base.nodes
            self.bounds = base.bounds
            self.fixed = base.fixed
            self.view = base.view
            self.evaluation_noise = base.evaluation_noise
            self._base = base
            self._weighed: dict = {}
        if tuple(modes):
            self.prepare(modes)

    @classmethod
    def at_fit(
        cls,
        study: Any,
        fit: Any,
        *,
        population: int,
        over: Any = None,
        weight_priors: bool = True,
        chunk: Any = "auto",
        solver: Any = None,
        modes: tuple = _MODES,
    ) -> PopulationRoute:
        """The route of ``fit`` on ``study``: ``PopulationRoute(study, fit, ...)`` by name.

        The lanes, rungs and first steps are the fit's; the solver's fixed
        coordinates, constraints and gauges enter every member's system; and
        each member's system and trial cost are weighed by its own weights.
        The arguments are the constructor's.
        """
        return cls(
            study,
            fit,
            population=population,
            over=over,
            weight_priors=weight_priors,
            chunk=chunk,
            solver=solver,
            modes=modes,
        )

    def prepare(self, modes: tuple) -> Any:
        """Compile (or reuse) the study's programs of ``modes``; their report becomes ``prepared``."""
        self.prepared = self.study.prepare(modes=tuple(modes))
        return self.prepared

    def resize(self, population: int) -> None:
        """Send every later request with ``population`` members; for a route on the study itself.

        Every member's held rungs and first steps become those member 0 holds
        (the fit's, before any request). A route cut for its population
        (``chunk`` not ``None``) keeps the width it was cut for and is not
        resized.

        Raises:
            ConfigurationError: If ``population`` is not a positive integer.
            ValueError: If the route was cut for its population.
        """
        if (
            isinstance(population, bool)
            or not isinstance(population, (int, np.integer))
            or population < 1
        ):
            raise ConfigurationError(
                f"population= is the number of points of every request, a positive integer; got "
                f"{population!r}."
            )
        if self._cut:
            raise ValueError(
                "this route's study was cut for its population; build a new route for "
                f"{int(population)} members instead of resizing it."
            )
        self.population = int(population)
        self.rungs = np.tile(self.rungs[:1], (self.population, 1))
        self.first_steps = np.tile(self.first_steps[:1], (self.population, 1))

    # Requests ---------------------------------------------------------------------------

    def evaluate(
        self,
        mode: str,
        points: Any,
        *,
        active: Any = None,
        lanes: Any = None,
        rungs: Any = None,
        first_steps: Any = None,
        units: Any = None,
    ) -> Any:
        """One request of ``mode`` over the whole population; the members ``active`` take part.

        A request that names no ``rungs`` and no ``first_steps`` runs every
        unit at the rung its member and lane hold (and names no rung when
        all of them hold rung 0). After the request every lane a member
        solved at a higher rung holds that rung for that member, never a
        lower one.

        Args:
            mode: The mode of the request.
            points: ``(Q, n)``: row ``q`` is population member ``q``'s point,
                every row sent whether it takes part or not.
            active: ``(Q,)`` booleans, the members that take part; every
                member when ``None``.
            lanes: The lanes to run; every lane when ``None``.
            rungs: ``(active members, requested lanes)``, in canonical order;
                ``None`` takes the rungs held.
            first_steps: As ``rungs``, the rung-0 first steps the rungs scale.
            units: ``(Q, requested lanes)`` booleans, lanes in the order of
                ``ids``: the units evaluated; a unit left out has the status
                ``absent`` (``Request(units=)``). ``None`` evaluates every unit.

        Returns:
            The study's [Results][diff_soup.inference.engine.Results], whose
            ``members`` are the population indices of the members that took
            part.

        Raises:
            ValueError: If ``points`` is not of shape ``(Q, n)``, ``active``
                does not flag the ``Q`` members, or a lane is not a lane of
                the study.
        """
        points = np.asarray(points)
        if points.shape != (self.population, self.n):
            raise ValueError(
                f"points= holds one row per population member: shape ({self.population}, "
                f"{self.n}); got {points.shape}."
            )
        flags = np.ones((self.population,), dtype=bool)
        if active is not None:
            flags = np.asarray(active)
            if flags.dtype != bool or flags.shape != (self.population,):
                raise ValueError(
                    f"active= flags which of the {self.population} population members take "
                    f"part: booleans of shape ({self.population},); got {flags.dtype} of shape "
                    f"{flags.shape}."
                )
        rows = np.flatnonzero(flags)
        if lanes is None:
            positions = np.arange(len(self.ids))
        else:
            unknown = [str(lane) for lane in lanes if str(lane) not in self.position]
            if unknown:
                raise ValueError(
                    f"lanes= names {unknown[:8]}, which are not lanes of this study; its lanes "
                    f"are {list(self.ids[:8])}{' ...' if len(self.ids) > 8 else ''}."
                )
            positions = np.sort([self.position[str(lane)] for lane in lanes])
        if rungs is None and first_steps is None:
            held = self.rungs[np.ix_(rows, positions)]
            if np.any(held > 0):
                rungs = held.copy()
                first_steps = self.first_steps[np.ix_(rows, positions)].copy()
        request = Request(
            mode=mode,
            points=points,
            active=None if active is None else flags,
            lanes=lanes,
            rungs=rungs,
            first_steps=first_steps,
            units=units,
        )
        started = time.perf_counter()
        results = self.study.evaluate(request)
        seconds = time.perf_counter() - started
        _hold_rungs(self.rungs, self.first_steps, results, self.position)
        self.requests.append(
            MappingProxyType(
                {
                    "mode": str(mode),
                    "members": int(rows.shape[0]),
                    "lanes": int(positions.shape[0]),
                    "batches": results.batches,
                    "seconds": float(seconds),
                }
            )
        )
        return results

    # One member's objective --------------------------------------------------------------

    def _weighted(self, weights: Any, weight_priors: Any = None) -> _LaneRoute:
        """The fit's route weighed by one member's weights (kept for the next call).

        ``weight_priors`` is ``True`` or ``False`` to weigh the regularizer
        blocks so whatever the route's own setting; ``None`` takes that setting.
        """
        values = np.asarray(weights, dtype=np.float64)
        count = int(self.tables.counts[0])
        if values.shape != (count,):
            raise ValueError(
                f"weights= gives one weight per member of the level {self.over!r}, which has "
                f"{count} members; got weights of shape {values.shape}."
            )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                f"weights= are finite and at least 0; got {values.tolist()[:8]}"
                f"{' ...' if count > 8 else ''}."
            )
        priors = self.weight_priors if weight_priors is None else bool(weight_priors)
        key = (values.tobytes(), priors)
        route = self._weighed.get(key)
        if route is None:
            route = self._base.weighted((self.over, values), weight_priors=priors)
            if len(self._weighed) >= max(2 * self.population, 64):
                self._weighed.pop(next(iter(self._weighed)))
            self._weighed[key] = route
        return route

    def system(self, results: Any, point: Any, *, member: int, weights: Any) -> Any:
        """Population member ``member``'s level system at ``point`` under ``weights``.

        Args:
            results: A ``blocks`` result in which ``member`` took part.
            point: ``(n,)``, the member's point of that request.
            member: The population index of the member.
            weights: One weight per member of the top level.

        Returns:
            The level system: the member's lanes, regularizer blocks and
            gauge rows weighed by ``weights``; a member of weight zero and
            every node below it held.
        """
        return self._weighted(weights).system(results, np.asarray(point), member=int(member))

    def trial_cost(self, results: Any, point: Any, *, member: int, weights: Any) -> float:
        """Population member ``member``'s cost at ``point`` under ``weights``.

        ``sum_l w_l c_l + sum_b w_b 0.5 |r_b|^2``: its lanes' costs from a
        ``value`` result of every lane, weighed by their members' weights,
        and the regularizer's blocks at ``point``, each weighed by its
        member's weight (by one when ``weight_priors`` is ``False``).
        """
        return self._weighted(weights).trial_cost(results, np.asarray(point), member=int(member))

    def failures(self, results: Any, *, member: int, weights: Any) -> tuple:
        """``(counted, ignored)``: population member ``member``'s failed lanes under ``weights``.

        A failed lane of weight zero lies outside the member's objective: it
        is returned among the ignored ones and decides nothing.
        """
        records = self._base.failures(results)
        mine = tuple(
            record
            for failure, record in zip(results.failures(), records)
            if int(failure["member"]) == int(member)
        )
        return self._weighted(weights).weigh(mine)

    def _failures(self, results: Any) -> dict:
        """The failed lanes of ``results`` by population member, with their nodes and rungs; at a fit."""
        grouped: dict = {}
        records = self._base.failures(results)
        for failure, record in zip(results.failures(), records):
            grouped.setdefault(int(failure["member"]), []).append(record)
        return {member: tuple(entries) for member, entries in grouped.items()}


# ---------------------------------------------------------------------------------------
# The lockstep loop over a population's members
# ---------------------------------------------------------------------------------------


class _NotFinite(ArithmeticError):
    """A level system, or its solve, that is not finite."""


def _held_from(results: Any, member: int) -> tuple:
    """``(costs, rows, failed)`` of every lane of population member ``member`` in ``results``."""
    row = int(np.flatnonzero(np.asarray(results.members) == int(member))[0])
    costs = np.array(np.asarray(results.cost)[row], dtype=np.float64)
    rows = np.array(np.asarray(results.rows)[row], dtype=np.float64)
    failed = np.asarray(results.status.kind)[row] != 0
    return costs, rows, failed


class Lockstep:
    """A population's members moved together: one attempt of every member still running per iteration.

    A subclass gives the population: ``members`` (in population order, each
    with ``index``, ``point``, ``running``, ``seconds`` and ``fail(reason,
    error)``), ``route`` (the ``PopulationRoute`` every request goes through),
    ``events``, ``iteration`` and ``seconds`` (empty, 0 and 0 before the first
    ``run``), ``counts()`` (the outcomes so far) and ``iterate(running)``, one
    lockstep iteration over the members ``running``. ``run`` makes the
    iterations and, after each, records its event.
    """

    def _points(self) -> np.ndarray:
        """``(Q, n)``: every member's current point."""
        return np.stack([np.asarray(member.point, dtype=np.float64) for member in self.members])

    def _mask(self, chosen: list) -> np.ndarray:
        active = np.zeros((len(self.members),), dtype=bool)
        active[[member.index for member in chosen]] = True
        return active

    def _host(self, member: Any, work: Any) -> Any:
        """Run one member's host work, timed; an exception ends that member failed, not the run."""
        mark = time.perf_counter()
        try:
            return work()
        except Exception as error:  # noqa: BLE001 -- that member's outcome; the run goes on
            member.fail("system" if isinstance(error, _NotFinite) else "error", error)
            return None
        finally:
            member.seconds += time.perf_counter() - mark

    def run(self, callbacks: tuple = (), *, checkpoint: Any = None) -> None:
        """Make lockstep iterations until no member runs; one event per iteration.

        Iterations are numbered on from the last one made (a restored
        population continues the count it was saved at). After each
        iteration its event goes to ``checkpoint`` (a ``PopulationCheckpoint``,
        which logs it and writes the population's state when due), then each
        callback's ``on_step`` (or the callback itself, when it has none) is
        called with it; an exception a callback raises ends the run and
        propagates, the iteration's checkpoint already written.
        """
        route = self.route
        started = time.perf_counter()
        iteration = self.iteration
        while True:
            running = [member for member in self.members if member.running]
            if not running:
                break
            iteration += 1
            mark = time.perf_counter()
            first_request = len(route.requests)
            self.iterate(running)
            wall = time.perf_counter() - mark
            made = route.requests[first_request:]
            event = MappingProxyType(
                {
                    "iteration": iteration,
                    "active": len(running),
                    "requests": tuple(
                        MappingProxyType(
                            {
                                "mode": record["mode"],
                                "members": record["members"],
                                "lanes": record["lanes"],
                                "calls": int(
                                    sum(
                                        int(getattr(batch, "calls", 0))
                                        for batch in dict(record["batches"]).values()
                                    )
                                ),
                                "seconds": record["seconds"],
                            }
                        )
                        for record in made
                    ),
                    "seconds": wall,
                    "host_seconds": wall - sum(float(record["seconds"]) for record in made),
                    "counts": MappingProxyType(self.counts()),
                }
            )
            self.events.append(event)
            self.iteration = iteration
            if checkpoint is not None:
                checkpoint.record(self, event)
            for callback in callbacks:
                handler = getattr(callback, "on_step", None)
                (callback if handler is None else handler)(event)
        self.seconds = time.perf_counter() - started


# ---------------------------------------------------------------------------------------
# The checkpoint of a population's iterations
# ---------------------------------------------------------------------------------------

_CHECKPOINT_FORMAT = "diff_soup population checkpoint, version 1"
_CHECKPOINT_FILE = re.compile(r"population-(\d+)\.npz")
_EVENTS = "events.jsonl"
_PARTIAL = ".partial"


def _jsonable(value: Any) -> Any:
    """``value`` as numbers, strings, lists and mappings that JSON holds; another object as its text."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "_asdict"):
        return {str(key): _jsonable(item) for key, item in value._asdict().items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return str(value)


def declaration_differences(saved: Mapping, here: Mapping) -> list:
    """``(item, saved, here)`` for every item whose texts in ``saved`` and ``here`` differ.

    Items come in the order of ``saved``'s, then of the items only ``here`` has.
    """
    found = []
    for item in dict.fromkeys([*saved, *here]):
        one, other = saved.get(item), here.get(item)
        if one != other:
            found.append((item, one, other))
    return found


class PopulationCheckpoint:
    """A population's lockstep iterations, kept in one directory.

    After every ``every``-th iteration, and after the iteration at which the
    last member ends, one file ``population-NNNNNN.npz`` (``NNNNNN`` the
    iteration) holds the fingerprint of what the population was declared
    with, the iteration, every member's state (the stepper's ``state()``)
    and the products given at ``begin``. A file is written under a temporary
    name in the directory and renamed when complete, so a file of that name
    is always whole. ``keep="last"`` removes the older files once the new
    one is complete, so the directory holds one such file and the log;
    ``keep="all"`` keeps every one. ``events.jsonl`` gets one JSON line per
    iteration, the iteration's event.

    Attributes:
        directory: The directory.
        keep: ``"last"`` or ``"all"``.
        every: The iterations between two files.
        fingerprint: Item to text: what the population was declared with.
    """

    def __init__(self, directory: Any, *, keep: str, every: int, fingerprint: Mapping):
        """Open ``directory`` (made when missing) for a population declared with ``fingerprint``."""
        self.directory = Path(directory)
        self.keep = str(keep)
        self.every = int(every)
        self.fingerprint = {str(item): str(text) for item, text in fingerprint.items()}
        self.directory.mkdir(parents=True, exist_ok=True)
        self._products: dict = {"arrays": {}, "texts": {}}

    @property
    def events(self) -> Path:
        """The path of the log, one JSON line per iteration."""
        return self.directory / _EVENTS

    def files(self) -> list:
        """``(iteration, path)`` of every population file, oldest first."""
        found = []
        for path in self.directory.glob("population-*.npz"):
            match = _CHECKPOINT_FILE.fullmatch(path.name)
            if match is not None:
                found.append((int(match.group(1)), path))
        return sorted(found)

    def latest(self) -> Any:
        """The path of the newest population file, or ``None``."""
        found = self.files()
        return found[-1][1] if found else None

    def read(self, path: Any) -> dict:
        """What a population file holds: ``iteration``, ``fingerprint``, ``arrays``, ``texts``, ``products``.

        Raises:
            ValueError: If the file is not a whole population file of this format.
        """
        path = Path(path)
        try:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"]))
                arrays = {name: np.asarray(data[name]) for name in data.files if name != "meta"}
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as error:
            raise ValueError(
                f"{path} could not be read as a population checkpoint ({type(error).__name__}: "
                f"{error})."
            ) from error
        if not isinstance(meta, dict) or meta.get("format") != _CHECKPOINT_FORMAT:
            raise ValueError(f"{path} holds no {_CHECKPOINT_FORMAT!r} record.")
        return {
            "path": path,
            "iteration": int(meta["iteration"]),
            "fingerprint": {str(item): str(text) for item, text in meta["fingerprint"].items()},
            "arrays": {
                name[len("state.") :]: value
                for name, value in arrays.items()
                if name.startswith("state.")
            },
            "texts": meta["members"],
            "products": {
                "arrays": {
                    name[len("products.") :]: value
                    for name, value in arrays.items()
                    if name.startswith("products.")
                },
                "texts": meta["products"],
            },
        }

    def differences(self, fingerprint: Mapping) -> list:
        """``(item, saved, here)`` for every item the saved ``fingerprint`` gives another text."""
        return declaration_differences(fingerprint, self.fingerprint)

    def discard(self) -> None:
        """Remove every population file, every partial file and the log."""
        for _, path in self.files():
            path.unlink(missing_ok=True)
        for path in self.directory.glob(f"population-*{_PARTIAL}"):
            path.unlink(missing_ok=True)
        self.events.unlink(missing_ok=True)

    def begin(self, iteration: int, products: Mapping) -> None:
        """Start (``iteration`` 0) or resume after ``iteration``; ``products`` go into every file.

        A partial file left by an interrupted write is removed, and the log
        keeps the lines of the iterations up to ``iteration`` only, so that
        an iteration made again after the newest file is logged once.
        """
        self._products = {
            "arrays": dict(products.get("arrays", {})),
            "texts": dict(products.get("texts", {})),
        }
        for path in self.directory.glob(f"population-*{_PARTIAL}"):
            path.unlink(missing_ok=True)
        if int(iteration) <= 0 or not self.events.exists():
            self.events.unlink(missing_ok=True)
            return
        kept = []
        for line in self.events.read_text().splitlines():
            try:
                number = int(json.loads(line).get("iteration", 0))
            except (ValueError, TypeError, AttributeError):
                continue
            if number <= int(iteration):
                kept.append(line)
        partial = self.directory / f"{_EVENTS}{_PARTIAL}"
        partial.write_text("".join(f"{line}\n" for line in kept))
        os.replace(partial, self.events)

    def record(self, stepper: Any, event: Mapping) -> None:
        """Log one iteration's event; write the population's state when the iteration is due.

        ``stepper`` is the population's ``Lockstep``: its ``members[*].running``
        say whether the population ended, and its ``state()`` gives the
        ``(arrays, texts)`` the file holds.
        """
        with open(self.events, "a") as handle:
            handle.write(json.dumps(_jsonable(event)) + "\n")
        iteration = int(event["iteration"])
        ended = not any(member.running for member in stepper.members)
        if iteration % self.every == 0 or ended:
            arrays, texts = stepper.state()
            self.write(iteration, arrays, texts)

    def write(self, iteration: int, arrays: Mapping, texts: Mapping) -> Path:
        """Write the file of ``iteration`` whole, then, under ``keep="last"``, remove the older ones."""
        name = f"population-{int(iteration):06d}.npz"
        target = self.directory / name
        partial = self.directory / f"{name}{_PARTIAL}"
        meta = {
            "format": _CHECKPOINT_FORMAT,
            "iteration": int(iteration),
            "fingerprint": self.fingerprint,
            "members": _jsonable(texts),
            "products": _jsonable(self._products["texts"]),
        }
        payload = {f"state.{key}": np.asarray(value) for key, value in arrays.items()}
        payload.update(
            {
                f"products.{key}": np.asarray(value)
                for key, value in self._products["arrays"].items()
            }
        )
        payload["meta"] = np.asarray(json.dumps(meta))
        with open(partial, "wb") as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, target)
        if self.keep == "last":
            for number, path in self.files():
                if number != int(iteration):
                    path.unlink(missing_ok=True)
        return target
