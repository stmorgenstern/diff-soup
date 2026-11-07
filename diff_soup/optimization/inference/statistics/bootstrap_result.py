"""The result of a bootstrap: its draws, their outcomes and their summaries."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import numpy as np

from diff_soup.optimization.inference.statistics._bootstrap_saved import (
    _AtFit,
    _inverse_arrays,
    _plain,
    _saved_at_fit,
    _saved_decode,
    _SavedHandle,
    _SavedSchema,
)
from diff_soup.optimization.inference.statistics.bootstrap_intervals import (
    _SHARE,
    BootstrapBand,
    BootstrapInterval,
    BootstrapOneStep,
    _Context,
    _Draws,
    _grid,
    _part_refusal,
    _summarize,
)
from diff_soup.optimization.inference.statistics.bootstrap_outcomes import _KEPT, BootstrapOutcomes
from diff_soup.problem.parameters import ParameterEntry, ParameterOccurrence

_FORMAT = "diff_soup bootstrap result, version 1"


_OUTCOME_ARRAYS = (
    "index",
    "attempts",
    "accepted",
    "requests",
    "invalid_trials",
    "stationarity",
    "radius",
    "points",
    "final_cost",
    "data_cost",
    "data_rows",
    "training_rmse",
)


_OUTCOME_TEXTS = ("kind", "status", "reason", "stop", "error")


_ONE_STEP_ARRAYS = ("estimates", "jackknife", "laplace", "sandwich", "influence")


def _tupled(value: Any) -> Any:
    """Lists read back from JSON as tuples, at every depth."""
    if isinstance(value, list):
        return tuple(_tupled(item) for item in value)
    return value


def _npz(path: Any) -> str:
    """``path`` with ``.npz`` appended when missing."""
    return str(path) if str(path).endswith(".npz") else f"{path}.npz"


@dataclasses.dataclass(frozen=True, eq=False)
class BootstrapResult:
    """The replicates of a bootstrap run, with their intervals and bands.

    `boot[handle]` is `boot.values(handle)`. Save a result with `save`, read it back with
    `BootstrapResult.load`, and combine the parts of a sharded run with `BootstrapResult.merge`.
    A part alone answers only `values` and `laplace`.

    Attributes:
        weights: The weights each replicate gave the members, one row per replicate.
        members: The labels of the reweighted members, in the order of the columns of `weights`.
        over: The name of the reweighted level.
        weight_scheme: The `weights=` of the run, `"bayesian"` or `"multinomial"`.
        weight_priors: The `weight_priors=` of the run.
        steps: The `steps=` of the run, an integer or `"refit"`.
        max_attempts: The step attempts allowed per replicate, or `None` for `steps=0`.
        outcomes: How each replicate and each leave-one-member-out estimate ended, or `None` for
            `steps=0`.
        fit_params: The fit's point, where every replicate starts.
        fit_status: The status the bootstrapped fit ended with.
        fit_converged: Whether the bootstrapped fit converged.
        shard: The `(i, n)` of a sharded run's part, or `None` for a whole run.
    """

    weights: np.ndarray
    members: tuple
    over: str
    weight_scheme: str
    weight_priors: bool
    steps: Any
    max_attempts: Any
    one_step: BootstrapOneStep
    outcomes: Any
    fit_rmse: float
    fit_params: np.ndarray
    fit_status: str
    fit_converged: bool
    fit_stationarity: Any
    score_balance: float
    work: Mapping
    parameter_schema: Any
    _products: Any = dataclasses.field(repr=False)
    shard: Any = None
    replicate_indices: Any = None
    jackknife_indices: Any = None
    _counts: Any = dataclasses.field(default=None, repr=False)
    _declaration: Any = dataclasses.field(default=None, repr=False)

    def _whole(self, what: str) -> None:
        """Refuse, naming the merge, what a part does not give."""
        if self.shard is not None:
            raise ValueError(_part_refusal(self.shard, what))

    @property
    def counts(self) -> Any:
        """The replicates' outcomes, counted per status or status and reason; ``None`` at ``steps=0``.

        Raises:
            ValueError: If the result is a part of a sharded bootstrap.
        """
        self._whole("the count of the replicates' outcomes")
        return self._counts

    @property
    def replicates(self) -> int:
        """The number of replicates the result holds: ``R``, or a part's own."""
        return int(self.weights.shape[0])

    @property
    def n_members(self) -> int:
        """``M``, the members of the resampled level; with few members an interval may under-cover."""
        return len(self.members)

    @property
    def distinct_weights(self) -> int:
        """The number of distinct weight vectors among the ``R`` drawn.

        Under multinomial weights a few members admit few distinct vectors
        (3 members admit 10, 4 admit 35), so replicates repeat; under Bayesian
        weights every draw differs.
        """
        return int(np.unique(np.asarray(self.weights, dtype=np.float64), axis=0).shape[0])

    def _kept(self) -> np.ndarray:
        """``(R,)`` booleans: the replicates whose estimates enter the intervals."""
        if self.outcomes is None:
            return np.all(np.isfinite(self.one_step.estimates), axis=1)
        return self.outcomes.kept[: self.replicates]

    @property
    def n_kept(self) -> int:
        """The number of replicates kept for the intervals."""
        self._whole("the count of the kept replicates")
        return int(np.count_nonzero(self._kept()))

    @property
    def n_failed(self) -> int:
        """The number of replicates that failed, left out of the intervals."""
        self._whole("the count of the failed replicates")
        return self.replicates - self.n_kept

    @property
    def no_step(self) -> dict:
        """Per status and reason, the replicates that ended without an accepted step.

        Those that failed (``"failed: no_step"``, ...) and those kept although
        they never left the fit's point (``"converged"``: the first test of
        their stationarity ended them). Empty at ``steps=0``. Refused by a
        part.
        """
        self._whole("the count of the replicates without an accepted step")
        counts: dict = {}
        if self.outcomes is None:
            return counts
        outcomes = self.outcomes
        for q in range(self.replicates):
            if int(outcomes.accepted[q]) == 0:
                label = (
                    outcomes.status[q]
                    if outcomes.reason[q] is None
                    else f"{outcomes.status[q]}: {outcomes.reason[q]}"
                )
                counts[label] = counts.get(label, 0) + 1
        return counts

    @property
    def n_kept_without_step(self) -> int:
        """The kept replicates that ended without an accepted step, at the fit's point.

        Their estimate is the fit's point, which is not a draw; 0 at
        ``steps=0``. Refused by a part.
        """
        self._whole("the count of the kept replicates without an accepted step")
        if self.outcomes is None:
            return 0
        kept = self._kept()
        moved = np.asarray(self.outcomes.accepted[: self.replicates]) > 0
        return int(np.count_nonzero(kept & ~moved))

    def _entry(self, handle: Any) -> Any:
        """The schema entry of ``handle``; ``KeyError`` if it is not a parameter of the study."""
        try:
            return self.parameter_schema.entry(handle)
        except KeyError as error:
            raise KeyError(
                f"the handle {getattr(handle, 'name', handle)!r} is not a parameter of the "
                "bootstrapped study; give one of its Recipe parameter handles."
            ) from error

    def values(self, handle: Any, *, members: bool = False) -> Any:
        """Return a parameter's values at the kept replicates, one row per replicate.

        A row holds one value per member for a parameter of each member, and a function parameter
        gives functions in place of arrays. Under `weights="multinomial"` a member's value is NaN, or
        `None` for a function, in the replicates that did not draw it.

        Args:
            handle: A parameter handle of the bootstrapped study.
            members: Must be `True` for a parameter of each member.

        Returns:
            The values, stacked over the kept replicates.
        """
        entry = self._entry(handle)
        name = getattr(entry.handle, "name", getattr(handle, "name", handle))
        occurrences = tuple(entry.occurrences)
        shared = len(occurrences) == 1 and occurrences[0].owner is None
        if not shared and members is not True:
            raise ValueError(
                f"{name!r} is a parameter of each member, not of the shared coordinates: its "
                "Laplace covariance per member is boot.laplace(handle), and its bootstrap spread "
                "across the replicates is boot.values(handle, members=True)."
            )
        # Under multinomial weights a member a replicate did not draw holds the fit's values
        # there, which are not an estimate: those entries are NaN (None for a function).
        drawn = None
        if not shared and self.weight_scheme != "bayesian":
            position = {member: index for index, member in enumerate(self.members)}
            owners = [occurrence.owner for occurrence in occurrences]
            if len(set(owners)) != len(owners) or not all(owner in position for owner in owners):
                raise ValueError(
                    f"boot.values({name!r}, members=True) under weights={self.weight_scheme!r} "
                    f"needs a handle with one value per member of the level {self.over!r}, "
                    f"whose weights say which replicates drew it; {name!r} has values for "
                    f"{tuple(owners)!r}, and the members of {self.over!r} are "
                    f"{tuple(self.members)!r}. Ask for a handle of the level {self.over!r}, or "
                    "bootstrap with weights='bayesian'."
                )
            columns = np.asarray([position[owner] for owner in owners], dtype=np.int64)
            drawn = np.asarray(self.weights, dtype=np.float64)[self._kept()][:, columns] > 0.0
        if self.outcomes is None:
            points = np.asarray(self.one_step.estimates)[self._kept()]
        else:
            points = np.asarray(self.outcomes.points)[: self.replicates][self._kept()]
        if not points.shape[0]:
            raise ValueError(
                f"none of the {self.replicates} replicates was kept, so {name!r} has no "
                "estimate; boot.counts says how they ended."
            )
        function = getattr(entry.handle, "kind", None) == "function"

        def decoded(point: np.ndarray, occurrence: Any) -> Any:
            columns = np.asarray(occurrence.columns, dtype=np.int64)
            return entry.decode(jnp.asarray(point[columns]))

        if shared:
            values = [decoded(point, occurrences[0]) for point in points]
            if function:
                return tuple(values)
            return np.stack([np.asarray(value) for value in values], axis=0)
        rows = [[decoded(point, occurrence) for occurrence in occurrences] for point in points]
        if function:
            if drawn is not None:
                rows = [
                    [value if hit else None for value, hit in zip(row, hits)]
                    for row, hits in zip(rows, drawn)
                ]
            return tuple(tuple(row) for row in rows)
        stacked = np.stack([np.stack([np.asarray(value) for value in row]) for row in rows], axis=0)
        if drawn is None:
            return stacked
        hits = drawn.reshape(drawn.shape + (1,) * (stacked.ndim - 2))
        return np.where(hits, stacked, np.nan)

    def __getitem__(self, handle: Any) -> Any:
        """The kept replicates' estimates of a shared handle: ``boot.values(handle)``."""
        return self.values(handle)

    def laplace(self, handle: Any) -> np.ndarray:
        """Return a parameter's Laplace covariance at the fit.

        Args:
            handle: A parameter handle of the bootstrapped study.

        Returns:
            The covariance block, or one block per member, stacked, for a parameter of each member.
        """
        try:
            entry = self.parameter_schema.entry(handle)
        except KeyError as error:
            raise KeyError(
                f"the handle {getattr(handle, 'name', handle)!r} is not a parameter of the "
                "bootstrapped study; give one of its Recipe parameter handles."
            ) from error
        occurrences = tuple(entry.occurrences)
        blocks = [self._products.inverse_block(occurrence.columns) for occurrence in occurrences]
        if len(occurrences) == 1 and occurrences[0].owner is None:
            return blocks[0]
        return np.stack(blocks)

    def _draws(self) -> _Draws:
        """The kept replicates' end points, the leave-one-member-out end points, the counts."""
        self._whole("an interval or band")
        outcomes = self.outcomes
        if outcomes is None:
            return self.one_step._draws()
        replicates = self.replicates
        kept = np.asarray(outcomes.kept[:replicates], dtype=bool)
        failed: dict = {}
        for q in np.flatnonzero(~kept):
            label = f"{outcomes.status[q]}: {outcomes.reason[q]}"
            failed[label] = failed.get(label, 0) + 1
        jackknife_failed = tuple(
            (self.members[int(outcomes.index[q])], f"{outcomes.status[q]}: {outcomes.reason[q]}")
            for q in range(replicates, len(outcomes))
            if outcomes.status[q] not in _KEPT
        )
        unresolved = ""
        count = int(np.count_nonzero(kept))
        still = [q for q in np.flatnonzero(kept) if int(outcomes.accepted[q]) == 0]
        if count and len(still) > _SHARE * count:
            stops: dict = {}
            for q in still:
                stops[str(outcomes.stop[q])] = stops.get(str(outcomes.stop[q]), 0) + 1
            ended = ", ".join(f"{stop} {number}" for stop, number in stops.items())
            unresolved = (
                f"{len(still)} of the {count} kept replicates ({len(still) / count:.1%}) ended "
                f"without an accepted step, at the fit's point, which is not a draw (ended by: "
                f"{ended}); with more than 5 % of them the interval may be too narrow. Its "
                "endpoints are given."
            )
        return _Draws(
            points=np.asarray(outcomes.points, dtype=np.float64)[:replicates][kept],
            numbers=np.flatnonzero(kept),
            jackknife=np.asarray(outcomes.points, dtype=np.float64)[replicates:],
            jackknife_failed=jackknife_failed,
            replicates=replicates,
            failed=failed,
            unresolved=unresolved,
        )

    def interval(
        self, statistic: Any, *, level: float = 0.95, kind: str = "bca", members: bool = False
    ) -> BootstrapInterval:
        """Return the bootstrap interval of a parameter, or of a statistic of the parameters.

        The acceleration of `kind="bca"` comes from the leave-one-member-out estimates. A member's
        own parameter needs `members=True`, `weights="bayesian"` and `kind="percentile"`.

        Args:
            statistic: A parameter handle, or a function `statistic(view)` where `view[handle]` is
                that parameter's value at one estimate.
            level: The coverage, strictly between 0 and 1.
            kind: `"bca"` for the bias-corrected and accelerated interval, or `"percentile"`.
            members: Whether `statistic` reads a parameter of each member, as a dictionary from
                member label to value.

        Returns:
            A [`BootstrapInterval`][diff_soup.optimization.inference.BootstrapInterval].
        """
        fields = _summarize(
            self._context(), self._draws(), statistic, level=level, kind=kind, members=members
        )
        return BootstrapInterval(**fields)

    def band(
        self,
        statistic: Any,
        *,
        at: Any,
        level: float = 0.95,
        kind: str = "bca",
        members: bool = False,
    ) -> BootstrapBand:
        """Return the pointwise bootstrap band of a function parameter, or of a statistic, on a grid.

        Args:
            statistic: A function parameter handle, or a function `statistic(view, at)` returning
                values on the grid.
            at: The grid, a non-empty array of finite numbers.
            level: The coverage, strictly between 0 and 1.
            kind: `"bca"` for the bias-corrected and accelerated band, or `"percentile"`.
            members: Whether `statistic` reads a parameter of each member, as a dictionary from
                member label to value.

        Returns:
            A [`BootstrapBand`][diff_soup.optimization.inference.BootstrapBand].
        """
        fields = _summarize(
            self._context(),
            self._draws(),
            statistic,
            level=level,
            kind=kind,
            members=members,
            grid=_grid(at),
        )
        return BootstrapBand(**fields)

    def _context(self) -> _Context:
        """What every interval of this result reads besides its estimates."""
        return _Context(
            schema=self.parameter_schema,
            point=np.asarray(self.fit_params, dtype=np.float64),
            scheme=self.weight_scheme,
            members=tuple(self.members),
            fit_status=self.fit_status,
            fit_converged=bool(self.fit_converged),
        )

    @classmethod
    def _from_at_fit(cls, at: _AtFit, **fields: Any) -> BootstrapResult:
        """The result whose fit's point and one-step products are ``at``'s; ``fields`` are the rest.

        The one-step's intervals read the result's own context, and in a part
        they refuse as the result's do.
        """
        result = cls(
            one_step=BootstrapOneStep(
                **{name: getattr(at, name) for name in _ONE_STEP_ARRAYS},
                active_bounds=at.active_bounds,
                failures=at.failures,
            ),
            fit_rmse=at.fit_rmse,
            fit_params=at.point.copy(),
            score_balance=at.score_balance,
            _products=at.inverse,
            **fields,
        )
        one_step = dataclasses.replace(
            result.one_step, _context=result._context(), _part=result.shard
        )
        return dataclasses.replace(result, one_step=one_step)

    def save(self, path: Any) -> None:
        """Write the result to one ``.npz`` file, read back by ``BootstrapResult.load``.

        The file holds the weights, the one-step products, every population
        member's outcome, the counts, the fit's point and numbers, the
        columns of the inverse Gauss-Newton matrix solved so far (the shared
        coordinates', and any member's asked for by ``laplace``), the work
        report as plain numbers, text, lists and mappings, each parameter's
        name, kind, size and columns, and, for a part of a sharded bootstrap,
        its shard, the numbers of its replicates and members and the record of
        the settings the merge compares, so a saved part merges as it does in
        memory.

        Args:
            path: The destination file; ``.npz`` is appended when missing.

        Raises:
            TypeError: If the result's parameter schema does not list its
                entries (``parameter_schema.entries``), as the schema of a
                lowered study does.
        """
        entries = getattr(self.parameter_schema, "entries", None)
        if entries is None:
            raise TypeError(
                "this result's parameter schema does not list its entries "
                "(parameter_schema.entries), so a saved file could not decode a handle; save a "
                "result of inf.bootstrap on a lowered study, whose schema lists them."
            )
        point = np.asarray(self.fit_params, dtype=np.float64)
        schema = []
        for entry in entries:
            kind = getattr(entry.handle, "kind", None)
            record = {
                "name": str(entry.handle.name),
                "kind": None if kind is None else str(kind),
                "size": int(entry.size),
                "occurrences": [
                    [_plain(occurrence.owner), [int(column) for column in occurrence.columns]]
                    for occurrence in entry.occurrences
                ],
            }
            if kind != "function":
                columns = np.asarray(entry.occurrences[0].columns, dtype=np.int64)
                decoded = entry.decode(jnp.asarray(point[columns]))
                record["shape"] = [int(size) for size in np.shape(decoded)]
            schema.append(record)
        one = self.one_step
        meta = {
            "format": _FORMAT,
            "members": _plain(tuple(self.members)),
            "over": str(self.over),
            "weight_scheme": str(self.weight_scheme),
            "weight_priors": bool(self.weight_priors),
            "steps": self.steps,
            "max_attempts": self.max_attempts,
            "counts": _plain(self._counts),
            "shard": None if self.shard is None else [int(item) for item in self.shard],
            "declaration": None if self._declaration is None else dict(self._declaration),
            "fit_rmse": float(self.fit_rmse),
            "fit_status": str(self.fit_status),
            "fit_converged": bool(self.fit_converged),
            "fit_stationarity": self.fit_stationarity,
            "score_balance": float(self.score_balance),
            "schema": {"fingerprint": str(self.parameter_schema.fingerprint), "entries": schema},
            "one_step": {
                "active_bounds": _plain(one.active_bounds),
                "failures": _plain(one.failures),
            },
            "outcomes": None,
        }
        arrays = {
            "weights": np.asarray(self.weights, dtype=np.float64),
            "fit_params": point,
            "replicate_indices": np.asarray(self.replicate_indices, dtype=np.int64),
            "jackknife_indices": np.asarray(self.jackknife_indices, dtype=np.int64),
        }
        for name in _ONE_STEP_ARRAYS:
            arrays[f"one_step.{name}"] = np.asarray(getattr(one, name), dtype=np.float64)
        arrays.update(
            _inverse_arrays(getattr(self._products, "inverse_columns", {}) or {}, point.size)
        )
        outcomes = self.outcomes
        if outcomes is not None:
            texts = {name: _plain(getattr(outcomes, name)) for name in _OUTCOME_TEXTS}
            texts["replicates"] = int(outcomes.replicates)
            texts["active_bounds"] = _plain(outcomes.active_bounds)
            texts["failed_lanes"] = _plain(outcomes.failed_lanes)
            meta["outcomes"] = texts
            for name in _OUTCOME_ARRAYS:
                arrays[f"outcomes.{name}"] = np.asarray(getattr(outcomes, name))
        arrays["meta"] = np.asarray(json.dumps(meta))
        arrays["work"] = np.asarray(json.dumps(_plain(self.work)))
        np.savez(_npz(path), **arrays)

    @classmethod
    def load(cls, path: Any) -> BootstrapResult:
        """Read a result written by [save][diff_soup.optimization.inference.BootstrapResult.save].

        The loaded result answers ``values``, ``interval``, ``band`` and the
        one-step's ``interval`` and ``band`` as the saved one did, given the
        handles of the study the bootstrap was made on (a handle is found by
        its name; a function is decoded through the handle given).
        ``laplace`` answers for the columns the saved result had solved. The
        work report comes back as plain numbers, text, lists and mappings.

        Args:
            path: The file; ``.npz`` is appended when missing.

        Returns:
            The [BootstrapResult][diff_soup.optimization.inference.BootstrapResult].

        Raises:
            ValueError: If the file was not written by ``save``.
        """
        from diff_soup.optimization.inference.hierarchical._replicates import ReplicateLaneFailure

        target = _npz(path)
        with np.load(target, allow_pickle=False) as data:
            files = set(data.files)
            meta = json.loads(str(data["meta"])) if "meta" in files else {}
            if meta.get("format") != _FORMAT:
                raise ValueError(
                    f"{target} was not written by BootstrapResult.save: it holds no "
                    f"{_FORMAT!r} record. Load a file that BootstrapResult.save wrote."
                )
            work = json.loads(str(data["work"]))
            arrays = {
                name: np.asarray(data[name]) for name in files if name not in ("meta", "work")
            }
        entries = []
        for record in meta["schema"]["entries"]:
            handle = _SavedHandle(record["name"], record["kind"])
            entries.append(
                ParameterEntry(
                    handle=handle,
                    occurrences=tuple(
                        ParameterOccurrence(
                            owner=_tupled(owner), columns=tuple(columns), declarations=(handle,)
                        )
                        for owner, columns in record["occurrences"]
                    ),
                    size=int(record["size"]),
                    decode=_saved_decode(record["kind"], record.get("shape", ())),
                )
            )
        schema = _SavedSchema(meta["schema"]["fingerprint"], tuple(entries))
        members = _tupled(meta["members"])
        outcomes = None
        texts = meta["outcomes"]
        if texts is not None:
            outcomes = BootstrapOutcomes(
                replicates=int(texts["replicates"]),
                **{name: _tupled(texts[name]) for name in _OUTCOME_TEXTS},
                active_bounds=tuple(
                    tuple(tuple(int(item) for item in side) for side in member)
                    for member in texts["active_bounds"]
                ),
                failed_lanes=tuple(
                    tuple(
                        ReplicateLaneFailure(
                            lane=str(lane["lane"]),
                            kind=str(lane["kind"]),
                            t_reached=float(lane["t_reached"]),
                            rungs=tuple(int(rung) for rung in lane["rungs"]),
                            attempt=int(lane["attempt"]),
                            at=str(lane["at"]),
                            ignored=bool(lane["ignored"]),
                        )
                        for lane in member
                    )
                    for member in texts["failed_lanes"]
                ),
                **{name: arrays[f"outcomes.{name}"] for name in _OUTCOME_ARRAYS},
            )
        shard = meta.get("shard")
        shard = None if shard is None else (int(shard[0]), int(shard[1]))
        replicate_indices = arrays.get("replicate_indices")
        if replicate_indices is None:
            replicate_indices = np.arange(arrays["weights"].shape[0], dtype=np.int64)
        jackknife_indices = arrays.get("jackknife_indices")
        if jackknife_indices is None:
            jackknife_indices = np.arange(len(members), dtype=np.int64)
        declaration = meta.get("declaration")
        # The products of the evaluation at the fit, under the names a checkpoint file gives them.
        at = _saved_at_fit(
            {
                "arrays": {
                    **arrays,
                    "point": arrays["fit_params"],
                    **{name: arrays[f"one_step.{name}"] for name in _ONE_STEP_ARRAYS},
                },
                "texts": {**meta, **meta["one_step"]},
            }
        )
        stationarity = meta["fit_stationarity"]
        return cls._from_at_fit(
            at,
            weights=arrays["weights"],
            members=members,
            over=str(meta["over"]),
            weight_scheme=str(meta["weight_scheme"]),
            weight_priors=bool(meta["weight_priors"]),
            steps=meta["steps"],
            max_attempts=meta["max_attempts"],
            outcomes=outcomes,
            fit_status=str(meta["fit_status"]),
            fit_converged=bool(meta["fit_converged"]),
            fit_stationarity=None if stationarity is None else float(stationarity),
            work=MappingProxyType(work),
            parameter_schema=schema,
            shard=shard,
            replicate_indices=np.asarray(replicate_indices, dtype=np.int64),
            jackknife_indices=np.asarray(jackknife_indices, dtype=np.int64),
            _counts=meta["counts"],
            _declaration=None if declaration is None else dict(declaration),
        )

    @classmethod
    def merge(cls, parts: Any) -> BootstrapResult:
        """The full result of a bootstrap run in shards, from its parts.

        ``inf.bootstrap(..., shard=(i, n))`` runs the replicates ``r`` with
        ``r % n == i`` and the leave-one-member-out members ``m`` with
        ``m % n == i`` of one bootstrap, one process per shard on any device,
        and returns a part. Replicate ``r``'s weights depend on the key and
        ``r`` only, and its trust region on its weights only, so each
        replicate ends in its shard as it ends in the unsharded run. ``merge``
        takes the ``n`` parts, in any order, in memory or read back by
        [load][diff_soup.optimization.inference.BootstrapResult.load], and returns the
        result the unsharded run gives: the weights and every outcome at the
        replicate's or member's own number, the one-step estimates, the counts,
        and the intervals and bands taken over every replicate. The Laplace
        and sandwich covariances, the influence vectors and the fit's numbers
        are those of shard 0's part (every part computes them from the same
        evaluation at the fit). Its ``work`` lists every part's requests, and
        the parts' own reports under ``shards``.

        Args:
            parts: The parts of one bootstrap, one for each shard ``i`` of one
                ``n``.

        Returns:
            The [BootstrapResult][diff_soup.optimization.inference.BootstrapResult], with
            ``shard`` ``None``.

        Raises:
            ValueError: If ``parts`` is empty or holds a result that is not a
                part of a sharded bootstrap; the parts differ in the study,
                the fit's point, the key or a setting (each item that differs
                is named); they are shards of different counts ``n``; a shard
                is given more than once, or one is missing.
        """
        from diff_soup.optimization.inference.hierarchical._population import (
            declaration_differences,
        )

        parts = list(parts)
        if not parts:
            raise ValueError(
                "merge takes the parts of one bootstrap run with shard=(i, n); got none."
            )
        for part in parts:
            if not isinstance(part, BootstrapResult) or part.shard is None:
                what = (
                    "a full result (shard=None)"
                    if isinstance(part, BootstrapResult)
                    else f"a {type(part).__name__}"
                )
                raise ValueError(
                    f"merge takes the parts of one bootstrap run with shard=(i, n); got {what}."
                )
            if part._declaration is None:
                raise ValueError(
                    f"the part of shard {tuple(part.shard)} holds no record of the settings it "
                    "was run with, so it cannot be told to belong with the others; merge parts "
                    "returned by inf.bootstrap(..., shard=(i, n)) or read by load."
                )
        first = parts[0]
        found: dict = {}
        for part in parts[1:]:
            for item, one, other in declaration_differences(first._declaration, part._declaration):
                if item not in found:
                    found[item] = (
                        f"{item}: {one} in shard {tuple(first.shard)}, {other} in shard "
                        f"{tuple(part.shard)}"
                    )
        if found:
            raise ValueError(
                f"the parts are not shards of one bootstrap: they differ in {', '.join(found)} "
                f"({'; '.join(found.values())}). Merge the parts of one bootstrap: the same "
                "study, fit, key and settings, each part run with shard=(i, n) for one n."
            )
        counts = sorted({int(part.shard[1]) for part in parts})
        if len(counts) > 1:
            listed = ", ".join(f"shard {tuple(part.shard)}" for part in parts)
            raise ValueError(
                f"the parts are shards of different counts n ({listed}); merge the parts of one "
                "run in n shards: shard=(i, n) for i = 0, ..., n - 1, with one n."
            )
        n = counts[0]
        held: dict = {}
        for part in parts:
            i = int(part.shard[0])
            if i in held:
                raise ValueError(
                    f"the part of shard ({i}, {n}) is given more than once; give each of the {n} "
                    "parts once."
                )
            held[i] = part
        missing = [i for i in range(n) if i not in held]
        if missing:
            listed = ", ".join(f"({i}, {n})" for i in missing)
            raise ValueError(
                f"the part(s) of shard(s) {listed} are missing; merge all {n} parts of the "
                "bootstrap."
            )
        ordered = [held[i] for i in range(n)]
        base = ordered[0]
        replicates = int(base._declaration["replicates"])
        count = len(base.members)
        for i, part in enumerate(ordered):
            mine = np.asarray(part.replicate_indices, dtype=np.int64)
            members = np.asarray(part.jackknife_indices, dtype=np.int64)
            if not (
                np.array_equal(mine, np.arange(i, replicates, n))
                and np.array_equal(members, np.arange(i, count, n))
            ):
                raise ValueError(
                    f"the part of shard ({i}, {n}) holds the replicates {mine.tolist()[:8]} and "
                    f"the members {members.tolist()[:8]}; a part of shard (i, n) holds those "
                    "whose number modulo n is i. Merge the parts as inf.bootstrap returned them."
                )
        n_coords = int(np.asarray(base.fit_params).size)
        weights = np.zeros((replicates, count))
        estimates = np.full((replicates, n_coords), np.nan)
        jackknife = np.full((count, n_coords), np.nan)
        failures = []
        for part in ordered:
            weights[part.replicate_indices] = np.asarray(part.weights, dtype=np.float64)
            estimates[part.replicate_indices] = np.asarray(part.one_step.estimates)
            jackknife[part.jackknife_indices] = np.asarray(part.one_step.jackknife)
            failures.extend(part.one_step.failures)
        failures.sort(key=lambda failure: (failure.kind != "replicate", int(failure.index)))
        one_step = dataclasses.replace(
            base.one_step,
            estimates=estimates,
            jackknife=jackknife,
            failures=tuple(failures),
            _part=None,
        )
        outcomes = counts = None
        if base.outcomes is not None:
            # Population row of the full result -> (part, its row): the replicates, then the
            # leave-one-member-out members, each part's rows its replicates, then its members.
            source = [(ordered[r % n], r // n) for r in range(replicates)]
            source += [
                (ordered[m % n], len(ordered[m % n].replicate_indices) + m // n)
                for m in range(count)
            ]
            columns = {}
            for name in (*_OUTCOME_TEXTS, "active_bounds", "failed_lanes"):
                columns[name] = tuple(getattr(part.outcomes, name)[row] for part, row in source)
            for name in _OUTCOME_ARRAYS:
                columns[name] = np.stack(
                    [np.asarray(getattr(part.outcomes, name))[row] for part, row in source]
                )
            outcomes = BootstrapOutcomes(replicates=replicates, **columns)
            counts = {}
            for q in range(replicates):
                status, reason = outcomes.status[q], outcomes.reason[q]
                label = status if reason is None else f"{status}: {reason}"
                counts[label] = counts.get(label, 0) + 1
        requests = tuple(
            request for part in ordered for request in tuple(part.work.get("requests", ()))
        )
        work = MappingProxyType(
            {
                "requests": requests,
                "blocks": sum(1 for request in requests if request["mode"] == "blocks"),
                "value": sum(1 for request in requests if request["mode"] == "value"),
                "shards": tuple(part.work for part in ordered),
            }
        )
        return dataclasses.replace(
            base,
            weights=weights,
            one_step=one_step,
            outcomes=outcomes,
            work=work,
            shard=None,
            replicate_indices=np.arange(replicates, dtype=np.int64),
            jackknife_indices=np.arange(count, dtype=np.int64),
            _counts=counts,
            _declaration=dict(base._declaration),
        )

    def __repr__(self) -> str:
        """A one-line summary."""
        part = "" if self.shard is None else f", shard={tuple(self.shard)}"
        return (
            f"BootstrapResult(replicates={self.replicates}, members={len(self.members)}, "
            f"over={self.over!r}, weights={self.weight_scheme!r}, steps={self.steps!r}{part}, "
            f"fit {self.fit_status})"
        )


__all__ = ["BootstrapResult"]
