"""Bootstrap intervals and bands, and how a statistic's draws are summarized into them."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

_KINDS = ("bca", "percentile")
# An interval is withheld when more than this share of the requested replicates failed, and
# flagged when more than this share of the kept ones never left the fit's point.
_SHARE = 0.05


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

    lower: float | np.ndarray
    upper: float | np.ndarray
    level: float
    kind: str
    status: str | np.ndarray
    reason: str
    n_kept: int
    n_failed: int
    members: int
    centre: float | np.ndarray

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

    lower: float | np.ndarray
    upper: float | np.ndarray
    level: float
    kind: str
    status: str | np.ndarray
    reason: str
    n_kept: int
    n_failed: int
    members: int
    centre: float | np.ndarray
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


def _name(handle: object) -> str:
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

    def __call__(self, handle: object) -> tuple:
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


def _of_handle(
    handle: object, entries: _Entries, grid: np.ndarray | None
) -> Callable[[_View], object]:
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

    def statistic(view: _View) -> object:
        value = view[handle]
        if isinstance(value, dict):
            parts = list(value.values())
            if grid is None:
                return np.stack([np.asarray(part, dtype=np.float64) for part in parts])
            return np.stack([np.asarray(part(grid), dtype=np.float64) for part in parts])
        return value if grid is None else value(grid)

    return statistic


def _statistic_of(
    statistic: Any, entries: _Entries, grid: np.ndarray | None
) -> Callable[[_View], object]:
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
    statistic: Callable[[_View], object],
    point: np.ndarray,
    entries: _Entries,
    members: bool,
    where: str,
    shape: tuple[int, ...] | None,
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
    grid: np.ndarray | None = None,
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


def _part_refusal(shard: tuple[int, int], what: str) -> str:
    """The refusal of something a part of a sharded bootstrap does not give."""
    i, n = (int(item) for item in shard)
    return (
        f"this result is the part of shard ({i}, {n}) of a bootstrap run in {n} shards and holds "
        f"only that shard's replicates; {what} is taken over every replicate, so ask the full "
        f"result: inf.BootstrapResult.merge(parts) of the {n} parts."
    )


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
    [BootstrapResult][diff_soup.optimization.inference.BootstrapResult] take theirs over
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
    _context: _Context | None = dataclasses.field(default=None, repr=False)
    _part: tuple[int, int] | None = dataclasses.field(default=None, repr=False)

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
        [BootstrapResult.interval][diff_soup.optimization.inference.BootstrapResult.interval];
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
        [BootstrapResult.band][diff_soup.optimization.inference.BootstrapResult.band]; the
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


def _one_of(value: object, options: tuple) -> bool:
    """Whether ``value`` is one of the strings ``options`` (an array is none of them)."""
    return isinstance(value, str) and value in options


__all__ = ["BootstrapInterval", "BootstrapBand", "BootstrapOneStep"]
