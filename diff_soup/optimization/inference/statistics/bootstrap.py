"""The bootstrap of a fit over the members of the top level of its study.

``bootstrap`` reweighs the members of a lowered study's top level (its
particles, say). From one evaluation at the fit it computes every
replicate's one-step estimate and the fit's Laplace and cluster-sandwich
covariances; with ``steps`` it then runs every replicate's own trust region
from the fit, all replicates together, and reports how each one ended. Its
result gives percentile and BCa intervals of a parameter, or of any
statistic of the decoded parameters, and pointwise bands of a function on a
grid, each with a status that says whether it can be trusted; it saves to
one ``.npz`` file and loads from it. A run keeps a checkpoint of its
iterations in a directory and resumes from it; it runs in shards, one
process per shard, whose parts merge into the full result.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import operator
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.inference.fitting.result import OptResult
from diff_soup.problem.parameters import ParameterEntry, ParameterOccurrence, parameter_decoder

# Smallest number of resampling clusters a cluster bootstrap can report from.
_MIN_CLUSTERS = 3


# ---------------------------------------------------------------------------
# The bootstrap of a lowered study over the members of its top level
# ---------------------------------------------------------------------------

_SCHEMES = ("bayesian", "multinomial")
# The statuses of a replicate whose estimate enters the intervals.
_KEPT = ("converged", "steps_reached", "stalled")
_KINDS = ("bca", "percentile")
# An interval is withheld when more than this share of the requested replicates failed, and
# flagged when more than this share of the kept ones never left the fit's point.
_SHARE = 0.05
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


@dataclasses.dataclass(frozen=True, eq=False)
class BootstrapInterval:
    """A bootstrap interval, returned by `BootstrapResult.interval`.

    The ends are NaN where `status` is not `"ok"` or `"unresolved"`, and `reason` says why.

    Attributes:
        lower: The lower end, a number or an array shaped like the statistic.
        upper: The upper end, shaped like `lower`.
        level: The coverage.
        kind: `"bca"` or `"percentile"`.
        status: `"ok"`, `"unresolved"`, `"too_many_failed"`, `"jackknife_failed"` or
            `"degenerate"`, shaped like `lower`.
        reason: Sentences explaining any status other than `"ok"`, and any caveat, or an empty
            string.
        n_kept: The number of kept replicates.
        n_failed: The number of failed replicates.
        members: The number of reweighted members.
        centre: The statistic at the fit's point.
    """

    lower: Any
    upper: Any
    level: float
    kind: str
    status: Any
    reason: str
    n_kept: int
    n_failed: int
    members: int
    centre: Any

    def __repr__(self) -> str:
        """A one-line summary."""
        return (
            f"BootstrapInterval(lower={self.lower!r}, upper={self.upper!r}, level={self.level}, "
            f"kind={self.kind!r}, status={self.status!r}, n_kept={self.n_kept})"
        )


@dataclasses.dataclass(frozen=True, eq=False)
class BootstrapBand:
    """A pointwise bootstrap band on a grid, returned by `BootstrapResult.band`.

    It has every attribute of `BootstrapInterval`, with `lower`, `upper`, `status` and `centre`
    given per grid point.

    Attributes:
        at: The grid.
        draws: The statistic on the grid at each kept replicate, one row per replicate.
    """

    lower: Any
    upper: Any
    level: float
    kind: str
    status: Any
    reason: str
    n_kept: int
    n_failed: int
    members: int
    centre: Any
    at: np.ndarray
    draws: np.ndarray

    def __repr__(self) -> str:
        """A one-line summary."""
        statuses = sorted({str(status) for status in np.ravel(np.asarray(self.status))})
        return (
            f"BootstrapBand(points={np.asarray(self.at).size}, level={self.level}, "
            f"kind={self.kind!r}, statuses={statuses}, n_kept={self.n_kept})"
        )


class _Context(NamedTuple):
    """What every interval of one bootstrap reads besides its estimates."""

    schema: Any
    point: np.ndarray
    scheme: str
    members: tuple
    fit_status: str
    fit_converged: bool


class _Draws(NamedTuple):
    """The estimates one interval is taken over, and what qualifies them."""

    points: np.ndarray  # (K, n): the kept estimates
    numbers: np.ndarray  # (K,): their replicates' numbers
    jackknife: np.ndarray  # (M, n): the estimates that each leave one member out
    jackknife_failed: tuple  # (member label, how it ended) of each that failed
    replicates: int
    failed: dict  # how the failed replicates ended: count
    unresolved: str  # the reason an interval is flagged unresolved, or ""


def _name(handle: Any) -> str:
    """A handle's name, for messages."""
    return str(getattr(handle, "name", handle))


def _member_refusal(name: str) -> str:
    """The refusal of a handle of the members' own coordinates read without ``members=True``."""
    return (
        f"{name!r} is a parameter of each member, not of the shared coordinates: its Laplace "
        "covariance per member is boot.laplace(handle), and its bootstrap spread across the "
        "replicates is asked for with members=True (under weights='bayesian', "
        "kind='percentile')."
    )


class _Entries:
    """The schema's entries of the handles a statistic reads, each looked up once."""

    def __init__(self, schema: Any):
        """Look handles up in ``schema``; ``read_members`` records a member handle read."""
        self.schema = schema
        self.cache: dict = {}
        self.read_members = False

    def __call__(self, handle: Any) -> tuple:
        found = self.cache.get(id(handle))
        if found is None:
            try:
                entry = self.schema.entry(handle)
            except KeyError as error:
                raise KeyError(
                    f"the handle {_name(handle)!r} is not a parameter of the bootstrapped study; "
                    "give one of its Recipe parameter handles."
                ) from error
            occurrences = tuple(entry.occurrences)
            shared = len(occurrences) == 1 and occurrences[0].owner is None
            columns = tuple(np.asarray(item.columns, dtype=np.int64) for item in occurrences)
            # The handle is kept so that its id names it while the cache lives.
            found = (entry, shared, columns, handle)
            self.cache[id(handle)] = found
        return found


class _View:
    """One point's decoded parameters: ``view[handle]`` is what a fit's result gives for it.

    A shared handle gives its value (a callable law for a function handle). A
    handle of the members' own coordinates gives a mapping from member to
    value, and only when the members' coordinates were asked for.
    """

    __slots__ = ("_entries", "_members", "_point")

    def __init__(self, point: np.ndarray, entries: _Entries, members: bool):
        """The view of ``point``; ``members`` admits the members' own coordinates."""
        self._point = point
        self._entries = entries
        self._members = members

    def __getitem__(self, handle: Any) -> Any:
        entry, shared, columns, _ = self._entries(handle)
        if shared:
            return entry.decode(jnp.asarray(self._point[columns[0]]))
        if not self._members:
            raise ValueError(_member_refusal(_name(entry.handle)))
        self._entries.read_members = True
        return {
            occurrence.owner: entry.decode(jnp.asarray(self._point[part]))
            for occurrence, part in zip(entry.occurrences, columns)
        }


def _of_handle(handle: Any, entries: _Entries, grid: Any) -> Any:
    """The statistic a handle stands for: ``view[handle]``, or ``view[handle](grid)`` for a band."""
    entry, _, _, _ = entries(handle)
    name = _name(entry.handle)
    function = getattr(entry.handle, "kind", None) == "function"
    if grid is None and function:
        raise ValueError(
            f"{name!r} is a function: its bootstrap band on a grid is boot.band(handle, at=grid); "
            "an interval takes a value or a statistic."
        )
    if grid is not None and not function:
        raise ValueError(
            f"{name!r} is a value, not a function: its interval is boot.interval(handle); a band "
            "takes a function handle, or a statistic(view, grid)."
        )

    def statistic(view: _View) -> Any:
        value = view[handle]
        if isinstance(value, dict):
            parts = list(value.values())
            if grid is None:
                return np.stack([np.asarray(part, dtype=np.float64) for part in parts])
            return np.stack([np.asarray(part(grid), dtype=np.float64) for part in parts])
        return value if grid is None else value(grid)

    return statistic


def _statistic_of(statistic: Any, entries: _Entries, grid: Any) -> Any:
    """``statistic(view)``: a handle's own statistic, or the caller's callable."""
    if hasattr(statistic, "identity") or hasattr(statistic, "name"):
        return _of_handle(statistic, entries, grid)
    if not callable(statistic):
        raise TypeError(
            "the first argument is a Recipe parameter handle of the bootstrapped study, or a "
            f"callable statistic(view{'' if grid is None else ', grid'}); got "
            f"{type(statistic).__name__}."
        )
    if grid is None:
        return statistic
    return lambda view: statistic(view, grid)


def _value_at(
    statistic: Any, point: np.ndarray, entries: _Entries, members: bool, where: str, shape: Any
) -> np.ndarray:
    """The statistic at one point, checked: numbers, the fit's shape, finite."""
    value = statistic(_View(point, entries, members))
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"the statistic returned {type(value).__name__} at {where}; a statistic returns a "
            "number or an array of numbers."
        ) from error
    if shape is not None and array.shape != shape:
        raise ValueError(
            f"the statistic returned an array of shape {array.shape} at {where} and of shape "
            f"{shape} at the fit's point; a statistic returns one shape at every point."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"the statistic is not finite at {where}; it has to be finite at the fit's point, at "
            "every kept replicate and at every estimate that leaves one member out."
        )
    return array


def _member_rules(context: _Context, kind: str) -> None:
    """A member's own coordinates: a spread under Bayesian weights, a percentile interval only."""
    if context.scheme != "bayesian":
        raise ValueError(
            "a member's own coordinates have a bootstrap interval only under weights='bayesian'; "
            f"under weights={context.scheme!r} a member of weight zero stays at the fit's values, "
            "which are not an estimate. Their Laplace covariance per member is "
            "boot.laplace(handle); their values in the replicates that drew the member are "
            "boot.values(handle, members=True), NaN in the others; for an interval, bootstrap "
            "with weights='bayesian' and ask with members=True."
        )
    if kind != "percentile":
        raise ValueError(
            "a member's own coordinates have no BCa interval: an estimate that leaves a member out "
            "holds that member's coordinates at the fit, so it says nothing about them. Ask with "
            "kind='percentile' and members=True; a member's own uncertainty at the fit is its "
            "Laplace covariance, boot.laplace(handle)."
        )


def _check_options(level: Any, kind: Any, members: Any) -> float:
    """``level`` as a float, after refusing a level, kind or members setting out of range."""
    if (
        isinstance(level, bool)
        or not isinstance(level, (int, float, np.integer, np.floating))
        or not np.isfinite(level)
        or not 0.0 < float(level) < 1.0
    ):
        raise ValueError(
            f"level= is the coverage, a number strictly between 0 and 1; got {level!r}."
        )
    if not _one_of(kind, _KINDS):
        raise ValueError(f"kind= is 'bca' or 'percentile'; got {kind!r}.")
    if members is not True and members is not False:
        raise ValueError(f"members= is True or False; got {members!r}.")
    return float(level)


def _percentile(column: np.ndarray, level: float) -> tuple:
    """The quantiles at ``(1 - level) / 2`` and ``(1 + level) / 2``, linearly interpolated."""
    alpha = 1.0 - level
    lower, upper = np.quantile(column, (alpha / 2.0, 1.0 - alpha / 2.0))
    return float(lower), float(upper)


def _bca(column: np.ndarray, centre: float, jackknife: np.ndarray, level: float) -> tuple:
    """``(lower, upper, cause)``: the BCa interval of one entry; NaN and a cause when degenerate.

    The bias correction ``z0 = Phi^-1((#{below} + #{equal} / 2) / K)`` over
    the ``K`` kept estimates; the acceleration ``a = sum d^3 / (6 (sum
    d^2)^(3/2))`` with ``d`` the mean of the leave-one-member-out values less
    each one (``a = 0`` when ``sum d^2 = 0``); the levels
    ``Phi(z0 + (z0 + z) / (1 - a (z0 + z)))`` at ``z = Phi^-1((1 - level) /
    2)`` and ``Phi^-1((1 + level) / 2)``. ``cause`` is ``"one_side"`` when
    ``z0`` is infinite, ``"levels"`` when a level falls outside (0, 1).
    """
    from scipy.special import ndtr, ndtri

    alpha = 1.0 - level
    count = column.size
    below = int(np.sum(column < centre))
    equal = int(np.sum(column == centre))
    fraction = (below + 0.5 * equal) / count
    deviation = float(np.mean(jackknife)) - jackknife
    denominator = float(np.sum(deviation**2))
    a = float(np.sum(deviation**3)) / (6.0 * denominator**1.5) if denominator > 0.0 else 0.0
    if not 0.0 < fraction < 1.0:
        return float("nan"), float("nan"), "one_side"
    z0 = float(ndtri(fraction))
    levels = []
    for z in (float(ndtri(alpha / 2.0)), float(ndtri(1.0 - alpha / 2.0))):
        with np.errstate(divide="ignore", invalid="ignore"):
            shifted = np.float64(z0) + np.float64(z0 + z) / np.float64(1.0 - a * (z0 + z))
        levels.append(float(ndtr(shifted)))
    if not all(0.0 < value < 1.0 for value in levels):
        return float("nan"), float("nan"), "levels"
    lower, upper = np.quantile(column, levels)
    return float(lower), float(upper), None


def _listed(positions: list) -> str:
    """The first eight positions, for a message."""
    shown = ", ".join(str(position) for position in positions[:8])
    return shown + (" ..." if len(positions) > 8 else "")


def _summarize(
    context: _Context,
    draws: _Draws,
    statistic: Any,
    *,
    level: Any,
    kind: Any,
    members: Any,
    grid: Any = None,
) -> dict:
    """The fields of an interval (``grid`` None) or of a band of ``statistic`` over ``draws``."""
    level = _check_options(level, kind, members)
    entries = _Entries(context.schema)
    function = _statistic_of(statistic, entries, grid)
    centre = _value_at(function, context.point, entries, members, "the fit's point", None)
    if entries.read_members:
        _member_rules(context, kind)
    shape = centre.shape
    size = int(centre.size)
    replicates = draws.replicates
    failed = int(sum(draws.failed.values()))
    count = int(draws.points.shape[0])
    status = np.full((size,), "ok", dtype=object)
    lower = np.full((size,), np.nan)
    upper = np.full((size,), np.nan)
    values = np.zeros((0,) + shape)
    sentences = []
    if failed > _SHARE * replicates:
        status[:] = "too_many_failed"
        ended = ", ".join(f"{label} {number}" for label, number in draws.failed.items())
        sentences.append(
            f"{failed} of the {replicates} replicates failed ({ended}), more than 5 % of them, so "
            "the kept replicates may not represent the resampling and no interval is given."
        )
    else:
        rows = [
            _value_at(function, point, entries, members, f"replicate {int(number)}", shape)
            for point, number in zip(draws.points, draws.numbers)
        ]
        values = np.stack(rows) if rows else values
        if entries.read_members:
            _member_rules(context, kind)
        flat = values.reshape(count, size)
        if kind == "bca" and draws.jackknife_failed:
            status[:] = "jackknife_failed"
            which = ", ".join(
                f"the one that leaves out member {label!r} ({how})"
                for label, how in draws.jackknife_failed
            )
            sentences.append(
                "BCa takes its acceleration from the estimates that each leave one member out, "
                f"and {which} failed, so no BCa interval is given; the percentile interval, "
                "kind='percentile', is."
            )
        elif kind == "percentile":
            for j in range(size):
                lower[j], upper[j] = _percentile(flat[:, j], level)
        else:
            jackknife = np.stack(
                [
                    _value_at(
                        function,
                        point,
                        entries,
                        members,
                        f"the point that leaves member {label!r} out",
                        shape,
                    )
                    for point, label in zip(draws.jackknife, context.members)
                ]
            ).reshape(len(context.members), size)
            centres = centre.reshape(size)
            causes: dict = {}
            for j in range(size):
                lower[j], upper[j], cause = _bca(
                    flat[:, j], float(centres[j]), jackknife[:, j], level
                )
                if cause is not None:
                    status[j] = "degenerate"
                    causes.setdefault(cause, []).append(j)
            noun = "grid points" if grid is not None else "entries"
            for cause, positions in causes.items():
                text = (
                    "every kept estimate lies on one side of the fit's value, so the bias "
                    "correction of BCa is infinite"
                    if cause == "one_side"
                    else "the quantile levels BCa shifts to fall outside (0, 1)"
                )
                if shape == ():
                    text = f"{text[0].upper()}{text[1:]}"
                else:
                    text = (
                        f"At {len(positions)} of the {size} {noun} ({_listed(positions)}), {text}"
                    )
                sentences.append(
                    f"{text}; no BCa interval exists there, and the percentile interval, "
                    "kind='percentile', is given."
                )
        if draws.unresolved:
            status[status == "ok"] = "unresolved"
            sentences.append(draws.unresolved)
    if entries.read_members:
        sentences.append(
            "This is the spread that the uncertainty of the shared coordinates induces in each "
            "member's own coordinates across the replicates; it leaves out the noise of the "
            "member's own data. A member's own uncertainty at the fit is its Laplace covariance, "
            "boot.laplace(handle)."
        )
    if not context.fit_converged:
        sentences.append(
            f"The fit ended with status {context.fit_status!r} without converging, so every "
            "replicate carries the fit's own remaining step."
        )
    if shape == ():
        ends = (float(lower[0]), float(upper[0]), str(status[0]), float(centre))
    else:
        ends = (lower.reshape(shape), upper.reshape(shape), status.reshape(shape), centre)
    fields = {
        "lower": ends[0],
        "upper": ends[1],
        "level": level,
        "kind": str(kind),
        "status": ends[2],
        "reason": " ".join(sentences),
        "n_kept": count,
        "n_failed": failed,
        "members": len(context.members),
        "centre": ends[3],
    }
    if grid is not None:
        fields["at"] = grid
        fields["draws"] = values
    return fields


def _grid(at: Any) -> np.ndarray:
    """The grid of a band, refused unless it is one or more finite numbers."""
    grid = np.asarray(at)
    if grid.size == 0 or not np.issubdtype(grid.dtype, np.number) or not np.all(np.isfinite(grid)):
        raise ValueError(
            f"at= is the grid the band is evaluated on, one or more finite numbers; got {at!r}."
        )
    return grid


def _plain(value: Any) -> Any:
    """``value`` as numbers, strings, lists and mappings that JSON holds; another object as its text."""
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, jax.Array)):
        return np.asarray(value).tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "_asdict"):
        return {str(key): _plain(item) for key, item in value._asdict().items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    return str(value)


def _tupled(value: Any) -> Any:
    """Lists read back from JSON as tuples, at every depth."""
    if isinstance(value, list):
        return tuple(_tupled(item) for item in value)
    return value


def _npz(path: Any) -> str:
    """``path`` with ``.npz`` appended when missing."""
    return str(path) if str(path).endswith(".npz") else f"{path}.npz"


def _part_refusal(shard: Any, what: str) -> str:
    """The refusal of something a part of a sharded bootstrap does not give."""
    i, n = (int(item) for item in shard)
    return (
        f"this result is the part of shard ({i}, {n}) of a bootstrap run in {n} shards and holds "
        f"only that shard's replicates; {what} is taken over every replicate, so ask the full "
        f"result: inf.BootstrapResult.merge(parts) of the {n} parts."
    )


# The settings of the solver every replicate's trust region takes, in a bootstrap's fingerprint.
_SOLVER_SETTINGS = (
    "hessian",
    "x_scale",
    "radius_init",
    "radius_max",
    "eta",
    "grad_tol",
    "step_tol",
    "cost_tol",
    "cost_atol",
    "stationarity_tol",
    "noise_trial_limit",
    "noise_relaxed_ratio",
    "max_consecutive_invalid_trials",
    "fixed_indices",
    "constraints",
    "gauges",
)


def _digest(value: Any) -> str:
    """Sixteen hexadecimal characters of a digest of ``value``'s numbers and structure.

    Arrays enter by their type, shape and bytes, other leaves by their text,
    and the tree by its structure, so the digest is the same in every
    process for equal settings.
    """
    hasher = hashlib.sha256()
    leaves, tree = jax.tree_util.tree_flatten(value)
    hasher.update(str(tree).encode())
    for leaf in leaves:
        if isinstance(leaf, (np.ndarray, np.generic, jax.Array)):
            array = np.ascontiguousarray(np.asarray(leaf))
            hasher.update(f"{array.dtype.str}{array.shape}".encode())
            hasher.update(array.tobytes())
        else:
            hasher.update(repr(leaf).encode())
    return hasher.hexdigest()[:16]


def _key_text(key: Any) -> str:
    """A digest of a JAX random key's bits, the same for a typed key and its raw bits."""
    try:
        data = jax.random.key_data(key)
    except (TypeError, ValueError):
        data = key
    return _digest(np.asarray(data))


def _setting_text(value: Any) -> str:
    """A setting as text: a plain number, string or ``None`` as itself, anything else by its digest."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    return f"digest {_digest(value)}"


def _declaration(
    problem: Any,
    fit: Any,
    key: Any,
    *,
    scheme: str,
    replicates: int,
    steps: Any,
    max_attempts: Any,
    over: str,
    weight_priors: bool,
    solver: Any,
) -> dict:
    """What a bootstrap is declared with, item by item, as text: its fingerprint.

    The study's and its parameter schema's fingerprints, a digest of the
    fit's point (its bytes), a digest of the key, the weights, the
    replicates, the steps, the attempts, the level resampled, whether
    priors are weighed, and every setting of the solver the replicates'
    trust regions take. Two runs with equal fingerprints draw the same
    weights and start every replicate from the same state.
    """
    study = getattr(problem, "study", None)
    items = {
        "study": str(getattr(study, "fingerprint", None)),
        "parameter schema": str(getattr(problem.parameter_schema, "fingerprint", None)),
        "fit point": _digest(np.asarray(fit.params, dtype=np.float64)),
        "key": _key_text(key),
        "weights": str(scheme),
        "replicates": str(int(replicates)),
        "steps": str(steps),
        "max_attempts": str(max_attempts),
        "over": str(over),
        "weight_priors": str(bool(weight_priors)),
    }
    for name in _SOLVER_SETTINGS:
        items[f"solver {name}"] = _setting_text(getattr(solver, name, None))
    return items


def _differing(found: list) -> str:
    """``item A in the file, B here; ...`` for ``(item, A, B)`` triples."""
    return "; ".join(f"{item} {saved} in the file, {here} here" for item, saved, here in found)


class _AtFit(NamedTuple):
    """What a result keeps of the evaluation at the fit, for the replicates and members it holds."""

    point: np.ndarray
    estimates: np.ndarray
    jackknife: np.ndarray
    laplace: np.ndarray
    sandwich: np.ndarray
    influence: np.ndarray
    active_bounds: tuple
    failures: tuple
    fit_rmse: float
    score_balance: float
    inverse: Any  # inverse_columns and inverse_block of the Gauss-Newton matrix at the fit


def _at_fit(products: Any, jackknife: np.ndarray, lane_members: np.ndarray, count: int) -> _AtFit:
    """The evaluation at the fit's products, its leave-one-member-out rows those of ``jackknife``."""
    from diff_soup.inference.hierarchical._replicates import fit_training_rmse

    held = {int(member) for member in jackknife}
    return _AtFit(
        point=products.point.copy(),
        estimates=products.estimates,
        jackknife=products.jackknife[np.asarray(jackknife, dtype=np.int64)],
        laplace=products.laplace,
        sandwich=products.sandwich,
        influence=products.influence,
        active_bounds=tuple(products.active_bounds),
        failures=tuple(
            failure
            for failure in products.failures
            if failure.kind == "replicate" or int(failure.index) in held
        ),
        fit_rmse=fit_training_rmse(products, lane_members, count),
        score_balance=float(products.score_balance),
        inverse=products,
    )


def _inverse_arrays(solved: Mapping, size: int) -> dict:
    """The solved columns of the inverse Gauss-Newton matrix, ``(size,)`` each, as two arrays."""
    order = sorted(solved)
    return {
        "inverse.columns": np.asarray(order, dtype=np.int64),
        "inverse.values": (
            np.stack([np.asarray(solved[column], dtype=np.float64) for column in order])
            if order
            else np.zeros((0, size))
        ),
    }


def _at_fit_record(at: _AtFit) -> dict:
    """``{"arrays", "texts"}``: the products a checkpoint file keeps, to resume without the evaluation."""
    return {
        "arrays": {
            "point": np.asarray(at.point, dtype=np.float64),
            "estimates": np.asarray(at.estimates, dtype=np.float64),
            "jackknife": np.asarray(at.jackknife, dtype=np.float64),
            "laplace": np.asarray(at.laplace, dtype=np.float64),
            "sandwich": np.asarray(at.sandwich, dtype=np.float64),
            "influence": np.asarray(at.influence, dtype=np.float64),
            **_inverse_arrays(at.inverse.inverse_columns, np.asarray(at.point).size),
        },
        "texts": {
            "active_bounds": _plain(at.active_bounds),
            "failures": _plain(at.failures),
            "fit_rmse": float(at.fit_rmse),
            "score_balance": float(at.score_balance),
        },
    }


def _saved_at_fit(record: Mapping) -> _AtFit:
    """The products a checkpoint file kept, as the result keeps them."""
    from diff_soup.inference.hierarchical._replicates import OneStepFailure

    arrays, texts = record["arrays"], record["texts"]
    columns = arrays["inverse.columns"]
    return _AtFit(
        point=np.array(arrays["point"], dtype=np.float64, copy=True),
        estimates=arrays["estimates"],
        jackknife=arrays["jackknife"],
        laplace=arrays["laplace"],
        sandwich=arrays["sandwich"],
        influence=arrays["influence"],
        active_bounds=tuple(tuple(int(item) for item in side) for side in texts["active_bounds"]),
        failures=tuple(
            OneStepFailure(
                kind=str(item["kind"]),
                index=int(item["index"]),
                reason=str(item["reason"]),
                message=str(item["message"]),
            )
            for item in texts["failures"]
        ),
        fit_rmse=float(texts["fit_rmse"]),
        score_balance=float(texts["score_balance"]),
        inverse=_SavedInverse(
            {int(column): arrays["inverse.values"][row] for row, column in enumerate(columns)}
        ),
    )


def _resume(writer: Any, on_conflict: str) -> Any:
    """The newest population file's contents when this bootstrap wrote it, else ``None``.

    A file of another declaration (or one that cannot be read) is refused
    under ``on_conflict="error"``, naming every item that differs, and
    discarded with the log under ``"restart"``.
    """
    from diff_soup._errors import ConfigurationError

    path = writer.latest()
    if path is None:
        return None
    try:
        saved = writer.read(path)
    except ValueError as error:
        problem = str(error)
    else:
        found = writer.differences(saved["fingerprint"])
        if not found:
            return saved
        problem = (
            f"{path.name} was written by a bootstrap declared otherwise, its {', '.join(item for item, _, _ in found)} "
            f"differing ({_differing(found)})"
        )
    if on_conflict == "error":
        raise ConfigurationError(
            f"bootstrap checkpoint={str(writer.directory)!r} holds a run this bootstrap cannot "
            f"resume: {problem}. Resume with the settings the files were written with, pass "
            "on_conflict='restart' to discard them and start afresh, or give another directory."
        )
    writer.discard()
    return None


def _shard_of(shard: Any) -> Any:
    """``shard=`` as ``None`` or ``(i, n)``; refused unless ``0 <= i < n``, both integers."""
    from diff_soup._errors import ConfigurationError

    if shard is None:
        return None
    try:
        i, n = shard
    except (TypeError, ValueError):
        i = n = None
    whole = (
        isinstance(i, (int, np.integer))
        and not isinstance(i, bool)
        and _positive_integer(n)
        and 0 <= int(i) < int(n)
    )
    if not whole:
        raise ConfigurationError(
            f"bootstrap shard= is None or a pair (i, n) of integers with 0 <= i < n: shard i of "
            f"n runs the replicates whose number modulo n is i; got {shard!r}."
        )
    return int(i), int(n)


def _directory(checkpoint: Any) -> Any:
    """``checkpoint=`` as ``None`` or a path; refused unless a path, or a file is in the way."""
    from diff_soup._errors import ConfigurationError

    if checkpoint is None:
        return None
    try:
        path = Path(checkpoint)
    except TypeError:
        raise ConfigurationError(
            f"bootstrap checkpoint= is None or the path of a directory; got {checkpoint!r}."
        ) from None
    if path.exists() and not path.is_dir():
        raise ConfigurationError(
            f"bootstrap checkpoint= is a directory, and {path} is a file; give the path of a "
            "directory (it is made when missing)."
        )
    return path


class _SavedHandle(NamedTuple):
    """A parameter of a saved result, by its name and kind."""

    name: str
    kind: Any


class _SavedSchema:
    """The parameter schema of a loaded result: each parameter's name, kind, size and columns.

    A handle is found by its name (and, when two parameters share a name, its
    size). A value is decoded to the shape it had when the result was saved;
    a function is decoded by the handle given, which has to be the study's
    own function handle.
    """

    def __init__(self, fingerprint: str, entries: tuple):
        """The saved schema's fingerprint and its entries, in the schema's order."""
        self.fingerprint = fingerprint
        self.entries = entries

    @property
    def handles(self) -> tuple:
        """The saved parameters, by name and kind, in the schema's order."""
        return tuple(entry.handle for entry in self.entries)

    def entry(self, handle: Any) -> ParameterEntry:
        """The saved entry of the parameter ``handle`` names."""
        name = getattr(handle, "name", None)
        candidates = [entry for entry in self.entries if entry.handle.name == name]
        if len(candidates) > 1:
            size, _ = parameter_decoder(handle)
            candidates = [entry for entry in candidates if entry.size == size]
        if len(candidates) != 1:
            raise KeyError(
                f"parameter handle {name!r} is not one of the saved result's parameters "
                f"{[entry.handle.name for entry in self.entries]}."
            )
        saved = candidates[0]
        decode = saved.decode
        if saved.handle.kind == "function":
            size, decode = parameter_decoder(handle)
            if size != saved.size:
                raise KeyError(
                    f"the function handle {name!r} has {size} coefficients; the saved result's "
                    f"function {name!r} has {saved.size}. Give the study's own handle."
                )
        return ParameterEntry(
            handle=handle, occurrences=saved.occurrences, size=saved.size, decode=decode
        )


def _saved_decode(kind: Any, shape: Any) -> Any:
    """The decoder of a saved parameter: a value reshaped to its saved shape."""
    if kind == "function":

        def decode(values: Any) -> Any:
            raise TypeError(
                "a saved function is decoded through the study's own function handle: ask the "
                "loaded result with that handle."
            )

        return decode
    shape = tuple(int(size) for size in shape)
    return lambda values: jnp.reshape(jnp.asarray(values), shape)


class _SavedInverse:
    """The columns of the inverse Gauss-Newton matrix a saved result holds."""

    def __init__(self, columns: dict):
        """The saved columns, ``(n,)`` each, by coordinate."""
        self._columns = columns

    @property
    def inverse_columns(self) -> Any:
        """The saved columns, by coordinate."""
        return MappingProxyType(dict(self._columns))

    def inverse_block(self, columns: Any) -> np.ndarray:
        """The inverse on ``columns`` x ``columns``, from the saved columns."""
        columns = [int(column) for column in np.ravel(np.asarray(columns, dtype=np.int64))]
        missing = [column for column in columns if column not in self._columns]
        if missing:
            raise ValueError(
                f"this result holds the columns {sorted(self._columns)} of the inverse "
                f"Gauss-Newton matrix, not {missing[:8]}: a result read by load, or made by a "
                "run resumed from a checkpoint, holds the columns solved before it was saved "
                "(the shared coordinates', and those boot.laplace(handle) asked for). Ask "
                "boot.laplace(handle) before saving the result, or run the bootstrap without "
                "resuming."
            )
        if not columns:
            return np.zeros((0, 0))
        return np.stack([self._columns[column][columns] for column in columns], axis=1)


@dataclasses.dataclass(frozen=True, eq=False)
class BootstrapOneStep:
    """The one-step bootstrap: each replicate's full Gauss-Newton step from the fit.

    Everything here comes from one ``blocks`` evaluation at the fit's point;
    no lane is solved anywhere else. A replicate's estimate is the step a
    trust-region fit of its weighted objective takes from the fit's point when
    its radius does not cut the step (kept strictly inside the box when the
    study has bounds; fixed coordinates and members of weight zero do not
    move).

    Attributes:
        estimates: ``(R, n)``, one row per replicate; NaN rows for the
            replicates listed in ``failures``.
        jackknife: ``(M, n)``: row ``m`` is the one-step of the objective
            without member ``m`` (weight zero on ``m``, one elsewhere).
        laplace: ``(g, g)``, the Laplace covariance of the ``g`` shared
            coordinates: their block of ``H^-1``, ``H`` the Gauss-Newton
            matrix of the fit's own objective at its point.
        sandwich: ``(g, g)``, the cluster sandwich of the shared coordinates,
            ``sum_m u_m u_m^T`` over the members' influence vectors, without a
            small-sample factor.
        influence: ``(M, n)``: row ``m`` is ``u_m = H^-1 s_m``, ``s_m`` the
            gradient member ``m``'s lanes and its own regularizer rows add at
            the fit.
        active_bounds: The sides active at the fit's point,
            ``((index, -1 | 1), ...)``. The two matrices are those of the
            unconstrained quadratic model on these coordinates, while every
            one-step estimate stays inside the box.
        failures: The one-steps that could not be computed, each a record with
            ``kind`` (``"replicate"`` or ``"jackknife"``), ``index``,
            ``reason`` (``"gauge"``, ``"system"`` or ``"error"``) and
            ``message``.

    With equality rows (``solver=`` constraints or gauges) every matrix is
    the one on the rows' manifold. Fixed coordinates have zero rows and
    columns.

    ``interval`` and ``band`` take their intervals over the one-step
    estimates, with the leave-one-member-out one-steps as the jackknife of
    BCa, as those of the
    [BootstrapResult][diff_soup.inference.BootstrapResult] take theirs over
    the replicates' end points.

    In a part of a bootstrap run in shards, ``estimates`` holds the part's
    replicates and ``jackknife`` its leave-one-member-out members (in the
    order of the part's ``replicate_indices`` and ``jackknife_indices``),
    and ``interval`` and ``band`` are given by the merged result only.
    """

    estimates: np.ndarray
    jackknife: np.ndarray
    laplace: np.ndarray
    sandwich: np.ndarray
    influence: np.ndarray
    active_bounds: tuple
    failures: tuple
    _context: Any = dataclasses.field(default=None, repr=False)
    _part: Any = dataclasses.field(default=None, repr=False)

    def _draws(self) -> _Draws:
        """The one-step estimates that are finite, and the one-steps that could not be computed."""
        if self._part is not None:
            raise ValueError(_part_refusal(self._part, "an interval or band of the one-steps"))
        estimates = np.asarray(self.estimates, dtype=np.float64)
        replicates = int(estimates.shape[0])
        kept = np.all(np.isfinite(estimates), axis=1)
        failed: dict = {}
        for failure in self.failures:
            if failure.kind == "replicate":
                label = f"failed: {failure.reason}"
                failed[label] = failed.get(label, 0) + 1
        missing = replicates - int(np.count_nonzero(kept)) - sum(failed.values())
        if missing > 0:
            failed["failed: error"] = failed.get("failed: error", 0) + missing
        members = self._context.members
        return _Draws(
            points=estimates[kept],
            numbers=np.flatnonzero(kept),
            jackknife=np.asarray(self.jackknife, dtype=np.float64),
            jackknife_failed=tuple(
                (members[failure.index], f"failed: {failure.reason}")
                for failure in self.failures
                if failure.kind == "jackknife"
            ),
            replicates=replicates,
            failed=failed,
            unresolved="",
        )

    def interval(
        self, statistic: Any, *, level: float = 0.95, kind: str = "bca", members: bool = False
    ) -> BootstrapInterval:
        """A bootstrap interval over the one-step estimates.

        Takes the same arguments as
        [BootstrapResult.interval][diff_soup.inference.BootstrapResult.interval];
        the estimates are the one-step estimates, and BCa's acceleration
        comes from the leave-one-member-out one-steps.
        """
        fields = _summarize(
            self._context, self._draws(), statistic, level=level, kind=kind, members=members
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
        """A pointwise bootstrap band over the one-step estimates.

        Takes the same arguments as
        [BootstrapResult.band][diff_soup.inference.BootstrapResult.band]; the
        estimates are the one-step estimates.
        """
        fields = _summarize(
            self._context,
            self._draws(),
            statistic,
            level=level,
            kind=kind,
            members=members,
            grid=_grid(at),
        )
        return BootstrapBand(**fields)

    def __repr__(self) -> str:
        """A one-line summary."""
        return (
            f"BootstrapOneStep(replicates={self.estimates.shape[0]}, "
            f"members={self.jackknife.shape[0]}, shared={self.laplace.shape[0]}, "
            f"failures={len(self.failures)})"
        )


class BootstrapOutcome(NamedTuple):
    """How one replicate, or one leave-one-member-out member, ended.

    The fields are those of [BootstrapOutcomes][diff_soup.inference.statistics.bootstrap.BootstrapOutcomes]
    for one population member; ``point`` is its row of ``points``.
    """

    kind: str
    index: int
    status: str
    reason: Any
    stop: Any
    error: Any
    attempts: int
    accepted: int
    requests: int
    invalid_trials: int
    stationarity: float
    radius: float
    active_bounds: tuple
    failed_lanes: tuple
    point: np.ndarray
    final_cost: float
    data_cost: np.ndarray
    data_rows: np.ndarray
    training_rmse: float


@dataclasses.dataclass(frozen=True, eq=False)
class BootstrapOutcomes:
    """How every replicate, then every leave-one-member-out member, ended: one entry each.

    Population member ``q`` is replicate ``q`` for ``q < replicates``; member
    ``replicates + m`` leaves member ``m`` of the top level out (weight zero
    on it, one elsewhere) and serves the jackknife. Every attribute but
    ``replicates`` has one entry per population member, in that order;
    ``outcomes[q]`` gives member ``q``'s entries as one
    [BootstrapOutcome][diff_soup.inference.statistics.bootstrap.BootstrapOutcome],
    and iterating gives them all.

    Attributes:
        replicates: ``R``, the replicates requested.
        kind: ``"replicate"`` or ``"jackknife"``.
        index: The replicate's number, or the number of the member left out.
        status: ``"converged"`` (the fit's converged tests, or the test of
            the next attempt passed at the last step), ``"steps_reached"``
            (it made its accepted steps), ``"stalled"`` (it moved, then its
            radius collapsed, the evaluation noise ended it, invalid trials
            ended it, or its attempts ran out) or ``"failed"``.
            ``converged``, ``steps_reached`` and ``stalled`` members are kept:
            their points are estimates; ``failed`` ones are left out.
        reason: ``None`` for ``converged`` and ``steps_reached``. For
            ``stalled``: ``"radius_collapsed"``, ``"evaluation_noise"``,
            ``"invalid_trials"`` or ``"attempts"``. For ``failed``:
            ``"no_step"`` (it ended without converging and without an
            accepted step, at the fit's point, which is not a draw),
            ``"system"`` (its level system at a point, or its full step at the
            fit, was not finite), ``"gauge"`` (every member a gauge averages
            over weighs zero) or ``"error"`` (its host work raised; ``error``
            holds the message).
        stop: The event that ended it: ``"stationarity"``,
            ``"gradient_tolerance"``, ``"step_tolerance"``,
            ``"cost_tolerance"``, ``"evaluation_noise"``,
            ``"invalid_trials"``, ``"steps"`` (its accepted steps were
            reached), ``"attempts"`` (its attempts ran out), or the failure's
            reason.
        error: The type and message of the exception that ended it, or
            ``None``.
        attempts: The attempts it made.
        accepted: Its accepted steps.
        requests: The requests it took part in. The evaluation at the fit,
            one request of the whole bootstrap at one point, is counted for
            replicate 0 (in a sharded run, by the part that holds it); the
            ``blocks`` request that formed the running replicates' systems
            when a run resumed from a checkpoint is counted for each of them.
        invalid_trials: Its invalid trials, withdrawn acceptances included.
        stationarity: The stationarity number at its end point (the decrease
            its model predicts, relative to ``max(|cost|, 1)``).
        radius: Its trust-region radius at the end.
        active_bounds: The sides active at its end point,
            ``((index, -1 | 1), ...)``.
        failed_lanes: Every lane that failed at one of its trials or at a
            point it accepted, each a
            [ReplicateLaneFailure][diff_soup.inference.hierarchical._replicates.ReplicateLaneFailure]
            with ``lane``, ``kind``, ``t_reached``, ``rungs``, ``attempt``,
            ``at`` (``"trial"`` or ``"accepted"``) and ``ignored`` (a lane of
            weight zero).
        points: ``(Q, n)``, the point each member ended at.
        final_cost: ``(Q,)``, its weighted cost there, regularizer rows
            included; NaN for a member that failed before its first attempt.
        data_cost: ``(Q, M)``: per member of the top level, the unweighted
            data cost of its lanes there (half the squared whitened residual,
            no regularizer row); NaN for a member one of whose lanes, of
            weight zero, failed there.
        data_rows: ``(Q, M)``: per member of the top level, the residual rows
            of its lanes (NaN as ``data_cost``).
        training_rmse: ``(Q,)``, ``sqrt(sum_m w_m 2 C_m / sum_m w_m N_m)``
            over the members of nonzero weight: the root mean square whitened
            residual of the resampled data set, each member counted ``w_m``
            times.
    """

    replicates: int
    kind: tuple
    index: np.ndarray
    status: tuple
    reason: tuple
    stop: tuple
    error: tuple
    attempts: np.ndarray
    accepted: np.ndarray
    requests: np.ndarray
    invalid_trials: np.ndarray
    stationarity: np.ndarray
    radius: np.ndarray
    active_bounds: tuple
    failed_lanes: tuple
    points: np.ndarray
    final_cost: np.ndarray
    data_cost: np.ndarray
    data_rows: np.ndarray
    training_rmse: np.ndarray

    @property
    def kept(self) -> np.ndarray:
        """``(Q,)`` booleans: the members whose points are estimates (not ``failed``)."""
        return np.asarray([status in _KEPT for status in self.status], dtype=bool)

    def __len__(self) -> int:
        """``Q``, the replicates and the leave-one-member-out members."""
        return len(self.status)

    def __getitem__(self, member: int) -> BootstrapOutcome:
        """Population member ``member``'s entries."""
        q = operator.index(member)
        if not -len(self) <= q < len(self):
            raise IndexError(
                f"the outcomes hold {len(self)} population members; got member {member}."
            )
        return BootstrapOutcome(
            kind=self.kind[q],
            index=int(self.index[q]),
            status=self.status[q],
            reason=self.reason[q],
            stop=self.stop[q],
            error=self.error[q],
            attempts=int(self.attempts[q]),
            accepted=int(self.accepted[q]),
            requests=int(self.requests[q]),
            invalid_trials=int(self.invalid_trials[q]),
            stationarity=float(self.stationarity[q]),
            radius=float(self.radius[q]),
            active_bounds=self.active_bounds[q],
            failed_lanes=self.failed_lanes[q],
            point=self.points[q],
            final_cost=float(self.final_cost[q]),
            data_cost=self.data_cost[q],
            data_rows=self.data_rows[q],
            training_rmse=float(self.training_rmse[q]),
        )

    def __iter__(self) -> Iterator[BootstrapOutcome]:
        """Every population member's entries, in order."""
        return (self[q] for q in range(len(self)))

    def __repr__(self) -> str:
        """A one-line summary."""
        return (
            f"BootstrapOutcomes(replicates={self.replicates}, members={len(self)}, "
            f"kept={int(np.count_nonzero(self.kept[: self.replicates]))})"
        )


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
            A [`BootstrapInterval`][diff_soup.inference.BootstrapInterval].
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
            A [`BootstrapBand`][diff_soup.inference.BootstrapBand].
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
        """Read a result written by [save][diff_soup.inference.BootstrapResult.save].

        The loaded result answers ``values``, ``interval``, ``band`` and the
        one-step's ``interval`` and ``band`` as the saved one did, given the
        handles of the study the bootstrap was made on (a handle is found by
        its name; a function is decoded through the handle given).
        ``laplace`` answers for the columns the saved result had solved. The
        work report comes back as plain numbers, text, lists and mappings.

        Args:
            path: The file; ``.npz`` is appended when missing.

        Returns:
            The [BootstrapResult][diff_soup.inference.BootstrapResult].

        Raises:
            ValueError: If the file was not written by ``save``.
        """
        from diff_soup.inference.hierarchical._replicates import ReplicateLaneFailure

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
            entries.append(
                ParameterEntry(
                    handle=_SavedHandle(record["name"], record["kind"]),
                    occurrences=tuple(
                        ParameterOccurrence(owner=_tupled(owner), columns=tuple(columns))
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
        [load][diff_soup.inference.BootstrapResult.load], and returns the
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
            The [BootstrapResult][diff_soup.inference.BootstrapResult], with
            ``shard`` ``None``.

        Raises:
            ValueError: If ``parts`` is empty or holds a result that is not a
                part of a sharded bootstrap; the parts differ in the study,
                the fit's point, the key or a setting (each item that differs
                is named); they are shards of different counts ``n``; a shard
                is given more than once, or one is missing.
        """
        from diff_soup.inference.hierarchical._population import declaration_differences

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


def _draw_weights(
    key: Any, replicates: int, members: int, scheme: str, *, numbers: Any = None
) -> np.ndarray:
    """``(R, M)`` member weights: replicate ``r`` draws from ``jax.random.fold_in(key, r)``.

    ``"bayesian"``: ``M e / sum(e)`` with ``e`` independent exponentials
    (every weight positive, the weights summing to ``M``). ``"multinomial"``:
    the counts of ``M`` draws of a member with replacement. Replicate ``r``'s
    weights do not depend on the number of replicates. ``numbers`` draws the
    rows of those replicates only, in its order (a shard's).
    """
    rows = []
    chosen = range(replicates) if numbers is None else (int(number) for number in numbers)
    for replicate in chosen:
        try:
            sub = jax.random.fold_in(key, replicate)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"key= is a JAX random key, such as jax.random.key(seed); got {type(key).__name__}."
            ) from error
        if scheme == "multinomial":
            draws = np.asarray(jax.random.randint(sub, (members,), 0, members))
            rows.append(np.bincount(draws, minlength=members).astype(np.float64))
        else:
            raw = np.asarray(jax.random.exponential(sub, (members,)), dtype=np.float64)
            rows.append(members * raw / np.sum(raw))
    return np.stack(rows)


def _positive_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) >= 1


def _one_of(value: Any, options: tuple) -> bool:
    """Whether ``value`` is one of the strings ``options`` (an array is none of them)."""
    return isinstance(value, str) and value in options


def bootstrap(
    problem: Any,
    fit: Any,
    *,
    replicates: int,
    key: Any,
    over: Any = None,
    weights: str = "bayesian",
    steps: Any = 3,
    weight_priors: bool = True,
    solver: Any = None,
    max_attempts: Any = None,
    require_converged: bool = True,
    checkpoint: Any = None,
    keep: str = "last",
    every: int = 1,
    on_conflict: str = "error",
    callbacks: Any = (),
    chunk: Any = "auto",
    shard: Any = None,
) -> BootstrapResult:
    """Bootstrap a hierarchical fit by reweighting the members of the study's top level.

    Each replicate draws one weight per member and steps from the fit's point under those
    weights. The top level needs at least three members.

    Args:
        problem: The [`LoweredStudy`][diff_soup.inference.LoweredStudy] the fit was made on.
        fit: The [`OptResult`][diff_soup.inference.OptResult] of
            [`TrustRegion`][diff_soup.inference.TrustRegion] on `problem`.
        replicates: The number of replicates, a positive integer.
        key: The JAX random key that fixes the weights, such as `jax.random.key(0)`.
        over: The level to reweight, by name or as `1`, which must be the top level, or `None`
            for the top level.
        weights: `"bayesian"` for continuous weights that sum to the member count, or
            `"multinomial"` to resample the members with replacement.
        steps: The accepted trust-region steps each replicate takes from the fit, `0` for a
            one-step estimate, or `"refit"` to refit each replicate to convergence.
        weight_priors: Whether each member's prior and regularization terms take the member's
            weight.
        solver: The `TrustRegion` the replicates step with, which must not set `weights=`, or
            `None` for `TrustRegion(stationarity_tol=1e-8)`.
        max_attempts: The step attempts allowed per replicate, or `None` for the solver's
            `max_steps` under `"refit"` and `4 * steps` otherwise.
        require_converged: Whether to refuse a fit that did not converge.
        checkpoint: A directory to save the run in and resume it from, made when missing, used
            when `steps` is not `0`.
        keep: `"last"` to keep only the newest checkpoint file, or `"all"` to keep every one.
        every: The number of iterations between two checkpoint files.
        on_conflict: `"error"` to refuse a checkpoint saved with other settings, or `"restart"`
            to discard it.
        callbacks: Functions, or objects with `on_step`, called with a record of each iteration.
        chunk: The number of replicates evaluated together, `"auto"`, a positive integer, or
            `None` for the lowered study as it is.
        shard: A pair `(i, n)` that runs only the replicates whose number modulo `n` is `i`, or
            `None` for every replicate.

    Returns:
        A [`BootstrapResult`][diff_soup.inference.BootstrapResult].

    Raises:
        TypeError: If `problem` is not a lowered study or `fit` is not an `OptResult`.
        ValueError: If the fit is of another study, did not converge under
            `require_converged=True`, or the top level has fewer than three members.
    """
    from diff_soup._errors import ConfigurationError
    from diff_soup.inference import LoweredStudy
    from diff_soup.inference.hierarchical._population import PopulationCheckpoint, PopulationRoute
    from diff_soup.inference.hierarchical._replicates import (
        FitProducts,
        ReplicateStepper,
        inherited_noise,
    )
    from diff_soup.inference.hierarchical.level_solvers import TrustRegion

    if not isinstance(problem, LoweredStudy):
        raise TypeError(
            "bootstrap takes the lowered study the fit was made on, ds.lower(study); got "
            f"{type(problem).__name__}. Lower the study with ds.lower(study), fit it with "
            "inf.optsolve(lowered, inf.TrustRegion()), and bootstrap that fit."
        )
    if not isinstance(fit, OptResult):
        raise TypeError(
            f"bootstrap takes the result of inf.TrustRegion on the lowered study as fit=; got "
            f"{type(fit).__name__}."
        )
    if not _positive_integer(replicates):
        raise ConfigurationError(
            f"bootstrap replicates= is a positive integer; got {replicates!r}."
        )
    if not _one_of(weights, _SCHEMES):
        raise ConfigurationError(
            f"bootstrap weights= is 'bayesian' or 'multinomial'; got {weights!r}."
        )
    refit = _one_of(steps, ("refit",))
    if not (
        refit
        or (
            isinstance(steps, (int, np.integer)) and not isinstance(steps, bool) and int(steps) >= 0
        )
    ):
        raise ConfigurationError(
            f"bootstrap steps= is a number of accepted steps, an integer of at least 0, or "
            f"'refit'; got {steps!r}."
        )
    if weight_priors is not True and weight_priors is not False:
        raise ConfigurationError(
            f"bootstrap weight_priors= is True or False; got {weight_priors!r}."
        )
    if require_converged is not True and require_converged is not False:
        raise ConfigurationError(
            f"bootstrap require_converged= is True or False; got {require_converged!r}."
        )
    if max_attempts is not None and not _positive_integer(max_attempts):
        raise ConfigurationError(
            f"bootstrap max_attempts= is None or a positive integer; got {max_attempts!r}."
        )
    if not _one_of(keep, ("last", "all")):
        raise ConfigurationError(f"bootstrap keep= is 'last' or 'all'; got {keep!r}.")
    if not _positive_integer(every):
        raise ConfigurationError(f"bootstrap every= is a positive integer; got {every!r}.")
    if not _one_of(on_conflict, ("error", "restart")):
        raise ConfigurationError(
            f"bootstrap on_conflict= is 'error' or 'restart'; got {on_conflict!r}."
        )
    directory = _directory(checkpoint)
    part = _shard_of(shard)
    if part is not None and part[1] > int(replicates):
        raise ConfigurationError(
            f"bootstrap shard={shard!r} splits {int(replicates)} replicates into {part[1]} "
            "shards; every shard holds at least one replicate, so n is at most replicates=."
        )
    if not (chunk is None or _one_of(chunk, ("auto",)) or _positive_integer(chunk)):
        raise ConfigurationError(
            f"bootstrap chunk= is 'auto', a positive integer, or None; got {chunk!r}."
        )
    try:
        iter(callbacks)
    except TypeError as error:
        raise ConfigurationError(
            f"bootstrap callbacks= is a sequence of callbacks; got {type(callbacks).__name__}."
        ) from error

    state = getattr(fit, "state", None)
    saved = (
        None
        if state is None or "fingerprint" not in state
        else str(np.asarray(state["fingerprint"]))
    )
    if saved is not None and saved != str(problem.parameter_schema.fingerprint):
        raise ValueError(
            "fit= was made on another study than problem=; bootstrap the fit with the lowered "
            "study it was made on."
        )
    if require_converged and not fit.converged:
        raise ValueError(
            f"the fit ended with status {fit.status.value!r} and did not converge, so its point "
            "is not a stationary point of its objective and every replicate would carry the "
            "fit's own remaining step. Continue the fit until it converges, or pass "
            "require_converged=False to bootstrap from it as it is."
        )
    if solver is None:
        solver = TrustRegion(stationarity_tol=1e-8)
    looped = refit or int(steps) >= 1
    if looped and max_attempts is None:
        max_attempts = int(solver.max_steps) if refit else 4 * int(steps)
    population_route = functools.partial(
        PopulationRoute, problem, fit, over=over, weight_priors=weight_priors, solver=solver
    )
    # The evaluation at the fit prepares the study's own blocks programs when it is made; a run
    # resumed from a checkpoint makes none.
    home = population_route(population=1, chunk=None, modes=())
    members = tuple(home.members[0])
    count = len(members)
    if count < _MIN_CLUSTERS:
        raise ValueError(
            f"bootstrap resamples the members of the level {home.over!r}, and it has "
            f"{count}; resampling needs at least {_MIN_CLUSTERS} members."
        )
    # The replicates and leave-one-member-out members this run holds: every one, or a shard's.
    if part is None:
        numbers = np.arange(int(replicates), dtype=np.int64)
        jackknife = np.arange(count, dtype=np.int64)
    else:
        numbers = np.arange(part[0], int(replicates), part[1], dtype=np.int64)
        jackknife = np.arange(part[0], count, part[1], dtype=np.int64)
    drawn = _draw_weights(key, int(replicates), count, weights, numbers=numbers)
    lane_members = np.asarray(home.nodes, dtype=np.int64)[:, 0]
    declaration = _declaration(
        problem,
        fit,
        key,
        scheme=str(weights),
        replicates=int(replicates),
        steps="refit" if refit else int(steps),
        max_attempts=None if not looped else int(max_attempts),
        over=str(home.over),
        weight_priors=bool(weight_priors),
        solver=solver,
    )
    if not looped:
        home.prepare(("blocks",))
        products = FitProducts(home, fit, drawn, x_scale=solver.x_scale, numbers=numbers)
        at = _at_fit(products, jackknife, lane_members, count)
        requests = tuple(home.requests)
        work = {
            "prepared": home.prepared,
            "solves": products.solves,
            "seconds": products.seconds,
        }
        outcomes = counts = None
    else:
        # The replicates, then one member per member of the top level that leaves it out.
        population = int(numbers.size + jackknife.size)
        home.resize(population)
        writer = saved = None
        if directory is not None:
            writer = PopulationCheckpoint(
                directory,
                keep=keep,
                every=int(every),
                fingerprint={
                    **declaration,
                    "shard": "none" if part is None else f"({part[0]}, {part[1]})",
                },
            )
            saved = _resume(writer, on_conflict)
        products = None
        if saved is None:
            home.prepare(("blocks",))
            products = FitProducts(
                home, fit, drawn, x_scale=solver.x_scale, keep=True, numbers=numbers
            )
            at = _at_fit(products, jackknife, lane_members, count)
        else:
            at = _saved_at_fit(saved["products"])
        route = population_route(population=population, chunk=chunk, modes=("value", "blocks"))
        if products is not None:
            # Every member starts at the fit's point, so every member starts with the rungs and
            # first steps that the evaluation there left with the member that took part in it.
            evaluated = int(products.requested[0])
            route.rungs[...] = home.rungs[evaluated]
            route.first_steps[...] = home.first_steps[evaluated]
        stepper = ReplicateStepper(
            route,
            home,
            products,
            drawn,
            solver=solver,
            steps=steps,
            max_attempts=int(max_attempts),
            noise=inherited_noise(fit),
            numbers=numbers,
            jackknife=jackknife,
            state=saved,
            fit_converged=bool(getattr(fit, "converged", False)),
        )
        if writer is not None:
            writer.begin(stepper.iteration, _at_fit_record(at))
        if saved is not None:
            stepper.rebuild()
        stepper.run(tuple(callbacks), checkpoint=writer)
        columns = stepper.columns()
        host = columns.pop("host_seconds")
        outcomes = BootstrapOutcomes(replicates=int(numbers.size), **columns)
        counts = stepper.counts()
        requests = tuple(home.requests) + tuple(route.requests)
        attempts = int(np.sum(outcomes.attempts))
        solves = {} if products is None else dict(products.solves)
        seconds = {} if products is None else dict(products.seconds)
        work = {
            "prepared": route.prepared,
            "prepared_fit": home.prepared,
            "iterations": tuple(stepper.events),
            "solves": MappingProxyType(
                {**solves, "loop": int(sum(member.solves for member in stepper.members))}
            ),
            "seconds": MappingProxyType(
                {**seconds, "loop": stepper.seconds, "loop_host": float(np.sum(host))}
            ),
            "attempts": attempts,
            "host_seconds": host,
            "host_seconds_per_attempt": float(np.sum(host)) / attempts if attempts else 0.0,
        }
        if saved is not None:
            work["resumed_from"] = int(saved["iteration"])
    work = MappingProxyType(
        {
            "requests": requests,
            "blocks": sum(1 for request in requests if request["mode"] == "blocks"),
            "value": sum(1 for request in requests if request["mode"] == "value"),
            **work,
        }
    )
    stationarity = dict(fit.info or {}).get("stationarity")
    return BootstrapResult._from_at_fit(
        at,
        weights=drawn,
        members=members,
        over=str(home.over),
        weight_scheme=str(weights),
        weight_priors=bool(weight_priors),
        steps="refit" if refit else int(steps),
        max_attempts=None if not looped else int(max_attempts),
        outcomes=outcomes,
        fit_status=str(fit.status.value),
        fit_converged=bool(fit.converged),
        fit_stationarity=None if stationarity is None else float(stationarity),
        work=work,
        parameter_schema=problem.parameter_schema,
        shard=part,
        replicate_indices=numbers,
        jackknife_indices=jackknife,
        _counts=counts,
        _declaration=declaration,
    )


__all__ = [
    "BootstrapInterval",
    "BootstrapBand",
    "BootstrapOneStep",
    "BootstrapOutcome",
    "BootstrapOutcomes",
    "BootstrapResult",
    "bootstrap",
]
