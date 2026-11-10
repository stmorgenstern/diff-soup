"""An iteration of the nested fit's polish as two programs: one proposes every member's trial, one judges them.

``propose`` takes the system at the members' point and their state as arrays: the scale
and every member's subproblem coordinates, its gradient test, its solve at zero damping
(under its own rows), its stationarity and accuracy tests, and, while a member is left
and the polish has an iteration left, every active member's trust-region step under the
box and its rows, its trial point and the decrease its model predicts. ``judge`` takes
what the host learned at the members' combined trial (the predicate per member, the
members whose lanes failed, their costs) and gives every member's gain ratio,
acceptance, radius, growth lock, noise run, and step and cost tests. Between them the
host makes the trial's request, turns failures into records and re-solves the lanes
whose rung rose; after them it makes the requests for the moved members' matrices.

Both are compiled once per system shape and settings with ``jax.jit``; under inequality
rows the proposing program holds the rows and is kept on them (``RowSet.program``).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from diff_soup.optimization.inference.fitting.scales import XScale
from diff_soup.optimization.inference.hierarchical import _bounds as box
from diff_soup.optimization.inference.hierarchical import _inequality as inequality
from diff_soup.optimization.inference.hierarchical import _level_elimination as elimination
from diff_soup.optimization.inference.hierarchical._level_inner import (
    _inner_solve,
    _inner_step,
    _member_norms,
    _member_predicted,
    _member_sums,
    _node_model,
)
from diff_soup.optimization.inference.hierarchical._level_solve import _scaled
from diff_soup.optimization.inference.hierarchical._step_program import coordinates
from diff_soup.optimization.inference.hierarchical._trust_step import _bounded_gradient
from diff_soup.optimization.inference.hierarchical._verdicts import _within
from diff_soup.optimization.solvers._loop import (
    _NOISE_RUN,
    cost_small,
    cost_threshold,
    step_small,
)
from diff_soup.optimization.solvers._loop_judge import (
    _gain_ratio,
    _larger,
    _next_radius,
    _noise_dominated,
    cost_scale,
)

# Why a member's polish ended at an attempt: the codes of the record's ``inner_stop``.
NOT_POLISHED = 0  # no polish: the outer trial was not accepted, or the member holds no data
ACCURACY = 1  # its predicted decrease fell to inner_fraction of the outer step's
AT_MAX_STEPS = 2  # it made the polish's largest number of iterations
STEP_TOLERANCE = 3  # its step test
COST_TOLERANCE = 4  # its cost test
GRADIENT_TOLERANCE = 5  # its gradient test
STATIONARITY = 6  # its stationarity number at most the inner stationarity_tol
EVALUATION_NOISE = 7  # three noise-dominated rejected trials in a row
RADIUS = 8  # its radius fell to the smallest positive float
SKIPPED = 9  # the polish was switched off at this attempt
NO_FINITE_STEP = 10  # no finite step inside its radius: its subtree's model is not finite
UNSOLVED = 11  # its lanes failed at its polished point after every retreat


class PolishSettings(NamedTuple):
    """What the polish's programs take from the solver, fixed for a fit (static arguments).

    ``level`` is the cut level and ``count`` its number of members; ``fixed`` lists the
    fixed coordinates; ``grad_test`` whether the members' gradient test runs (the inner
    ``grad_tol`` set); the tolerances are the inner solver's; ``noise_limit`` is ``None``
    for no bound.
    """

    level: int
    count: int
    fixed: tuple
    grad_test: bool
    step_tol: float
    cost_tol: float
    cost_atol: float
    eta: float
    radius_max: float
    relaxed: bool
    noise_limit: float | None
    eps: float
    tiny: float


class Layout(NamedTuple):
    """Where every member's coordinates are in the joint vector.

    ``owners`` ``(n,)`` is the member of the cut level of every column (``-1`` above
    it); ``columns`` ``(N, W)`` every member's columns in order, padded with ``n``;
    ``slots`` ``(n,)`` every column's place among its member's.
    """

    owners: ArrayLike
    columns: ArrayLike
    slots: ArrayLike


def layout(owners: np.ndarray, count: int) -> Layout:
    """The ``Layout`` of the members whose column owners are ``owners``, as host arrays."""
    owners = np.asarray(owners, dtype=np.int64)
    owned = np.flatnonzero(owners >= 0)
    order = owned[np.argsort(owners[owned], kind="stable")]
    groups = owners[order]
    slot = np.arange(order.size) - np.searchsorted(groups, np.arange(count))[groups]
    width = int(np.bincount(groups, minlength=count).max(initial=1))
    columns = np.full((count, max(width, 1)), owners.size, dtype=np.int64)
    columns[groups, slot] = order
    slots = np.zeros(owners.shape, dtype=np.int64)
    slots[order] = slot
    return Layout(owners, columns, slots)


def _gathered(where: Layout, values: jax.Array, fill: float) -> jax.Array:
    """``(N, W)``: every member's entries of ``values``, ``fill`` in the padding."""
    padded = jnp.concatenate([values, jnp.full((1,), fill, dtype=values.dtype)])
    return padded[where.columns]


def _scattered(where: Layout, values: jax.Array, take: jax.Array, original: jax.Array):
    """``original`` with the columns of the members ``take`` read from ``values`` ``(N, W)``."""
    owners = where.owners
    member = jnp.where(owners >= 0, owners, 0)
    return jnp.where((owners >= 0) & take[member], values[member, where.slots], original)


class MemberBefore(NamedTuple):
    """The members at the start of a polish iteration, as arrays.

    ``scale`` is ``nan`` until the polish first sets it; ``costs`` every member's cost at
    ``point``; ``thresholds`` their gradient tests' thresholds (any number when the test
    is off); ``codes`` why each member's polish ended so far (``NOT_POLISHED``: not
    ended); ``level`` the gradient floor's level and ``accuracy`` ``inner_fraction``
    times the outer step's predicted decrease (``nan``: none); ``limit`` the members'
    stationarity yardstick; ``stepping`` whether the polish has an iteration left.
    """

    point: ArrayLike
    scale: ArrayLike
    costs: ArrayLike
    radii: ArrayLike
    thresholds: ArrayLike
    active: ArrayLike
    codes: ArrayLike
    level: ArrayLike
    accuracy: ArrayLike
    limit: ArrayLike
    stepping: ArrayLike


class MemberProposal(NamedTuple):
    """A polish iteration's proposal, and the members' numbers at its point.

    ``active`` and ``codes`` are after the tests at the point and after the step (a
    member with no finite step inside its radius ends with ``NO_FINITE_STEP``);
    ``stationarity`` is every member's number at the point; ``stepped`` whether a step
    was made, which it is only while a member is left and ``stepping``. ``trial``,
    ``step_norms``, ``scaled_norms`` and ``predicted`` are the members' trial and its
    norms in parameter units and in the subproblem's coordinates, and the decrease each
    member's model predicts; ``zero_solves`` the solves of the active set at zero
    damping beyond the first, and ``solves`` those of the step.
    """

    scale: ArrayLike
    active: ArrayLike
    codes: ArrayLike
    stationarity: ArrayLike
    stepped: ArrayLike
    trial: ArrayLike
    step_norms: ArrayLike
    scaled_norms: ArrayLike
    predicted: ArrayLike
    zero_solves: ArrayLike
    solves: ArrayLike


def _choose(
    system: elimination.LevelSystem,
    where: Layout,
    x: jax.Array,
    gradient: jax.Array,
    subproblem: box.Scaling,
    step: jax.Array,
    radii: jax.Array,
    active: jax.Array,
    predicted: jax.Array,
    bounds: tuple,
    level: int,
) -> tuple:
    """Every active member's trial step under the box, from its subproblem's solution.

    A member whose ``x + step`` stays in the box keeps its step and its predicted
    decrease; every other one takes ``_bounds.select_step`` on its scaled subtree model
    with its radius, one member per lane of a ``vmap``. Returns ``(step, scaled_step,
    predicted)`` in joint columns and per member.
    """
    lower, upper = bounds
    n = x.shape[0]
    d = 1.0 / subproblem.scale
    scaled = subproblem.scale * step
    g_h = d * gradient
    addition = subproblem.addition
    columns = jnp.asarray(where.columns)
    present = columns < n
    xs, ps, phs = (_gathered(where, value, 0.0) for value in (x, step, scaled))
    lows, highs = _gathered(where, lower, -np.inf), _gathered(where, upper, np.inf)
    reach = _larger(radii, jnp.linalg.norm(phs, axis=1))
    theta = jax.vmap(box.step_back_ratio)(
        _gathered(where, gradient, 0.0), _gathered(where, subproblem.v, 0.0)
    )

    def one(member, own, present_, x_, g_h_, p, p_h, d_, add, radius, low, high, theta_):
        def quadratic_form(vector):
            full = (
                jnp.zeros((n + 1,), dtype=x.dtype)
                .at[own]
                .add(jnp.where(present_, d_ * vector, 0.0))
            )
            _, quadratic = _node_model(system, full[:n], level=level)
            return quadratic[member] + add @ (vector * vector)

        return box.select_step(x_, quadratic_form, g_h_, p, p_h, d_, radius, low, high, theta_)

    chosen, chosen_scaled, value = jax.vmap(one)(
        jnp.arange(columns.shape[0]),
        columns,
        present,
        xs,
        _gathered(where, g_h, 0.0),
        ps,
        phs,
        _gathered(where, d, 1.0),
        _gathered(where, addition, 0.0),
        reach,
        lows,
        highs,
        theta,
    )
    take = active & ~jax.vmap(box.in_bounds)(xs + ps, lows, highs)
    return (
        _scattered(where, chosen, take, step),
        _scattered(where, chosen_scaled, take, scaled),
        jnp.where(take, value, predicted),
    )


def _shortened(
    where: Layout,
    x: jax.Array,
    step: jax.Array,
    gradient: jax.Array,
    subproblem: box.Scaling,
    holding: jax.Array,
    bounds: tuple,
) -> jax.Array:
    """``step`` with the members ``holding`` a row that leave the box shortened to stay inside.

    Such a member's step is cut to ``theta`` times the distance to the first side it
    reaches, which keeps its rows, as the joint step under equality rows does.
    """
    xs, ps = _gathered(where, x, 0.0), _gathered(where, step, 0.0)
    lows, highs = _gathered(where, bounds[0], -np.inf), _gathered(where, bounds[1], np.inf)
    inside = jax.vmap(box.in_bounds)(xs + ps, lows, highs)
    stride = jax.vmap(lambda *args: box.step_size_to_bound(*args)[0])(xs, ps, lows, highs)
    theta = jax.vmap(box.step_back_ratio)(
        _gathered(where, gradient, 0.0), _gathered(where, subproblem.v, 0.0)
    )
    factor = jnp.where(holding & ~inside, stride * theta, 1.0)
    owners = where.owners
    return jnp.where(owners >= 0, step * factor[jnp.where(owners >= 0, owners, 0)], step)


def _step(
    system: elimination.LevelSystem,
    where: Layout,
    x: jax.Array,
    gradient: jax.Array,
    subproblem: box.Scaling,
    addition: jax.Array,
    radii: jax.Array,
    active: jax.Array,
    first: tuple,
    own: inequality.MemberRowSet | None,
    bounds: tuple | None,
    settings: PolishSettings,
    rows: Any,
    owners: np.ndarray | None,
) -> tuple:
    """Every active member's trial: its step under its rows and the box, and the trial's norms.

    Returns ``(trial, step_norms, scaled_norms, predicted, no_step, solves)``, ``no_step``
    the members with no finite step inside their radius, which take none.
    """
    level, count = settings.level, settings.count
    scale = subproblem.scale
    member_of = where.owners
    if own is None:
        step, predicted, multipliers, solves, _ = _inner_step(
            system,
            scale,
            radii,
            active,
            addition,
            member_of,
            first,
            level=level,
            eps=settings.eps,
            tiny=settings.tiny,
        )
    else:
        step, predicted, multipliers, solves, held, holding = inequality.member_step(
            own,
            system,
            subproblem,
            x,
            radii,
            active,
            first,
            level,
            owners,
            eps=settings.eps,
            tiny=settings.tiny,
        )
        if bounds is not None:
            step = _shortened(where, x, step, gradient, subproblem, holding, bounds)
            model = _member_predicted(
                system, step, scale * step, addition, member_of, count, level=level
            )
            predicted = jnp.where(holding, model, predicted)
    no_step = active & jnp.isnan(multipliers)
    active = active & ~no_step
    if settings.fixed:
        step = step.at[np.asarray(settings.fixed)].set(0.0)
    moving = (member_of >= 0) & active[jnp.where(member_of >= 0, member_of, 0)]
    if bounds is None:
        trial = jnp.where(moving, x + step, x)
        scaled = scale * step
    else:
        step, scaled, predicted = _choose(
            system, where, x, gradient, subproblem, step, radii, active, predicted, bounds, level
        )
        inside = box.make_strictly_feasible(x + step, bounds[0], bounds[1], rstep=0)
        trial = jnp.where(moving, inside, x)
    if own is not None:
        trial, scaled, predicted = inequality.member_back(
            rows, own, system, x, trial, subproblem, predicted, held, level, owners, bounds
        )
    return (
        trial,
        _member_norms(trial - x, member_of, count),
        _member_norms(scaled, member_of, count),
        predicted,
        no_step,
        solves,
    )


def propose(
    system: elimination.LevelSystem,
    x_scale: XScale,
    bounds: tuple | None,
    where: Layout,
    before: MemberBefore,
    *,
    settings: PolishSettings,
    rows: Any = None,
    owners: np.ndarray | None = None,
) -> MemberProposal:
    """A polish iteration's proposal: every member's tests at its point, then its trial.

    In this order: the scale (the larger of ``before.scale`` and the system's
    candidate) and the subproblem's coordinates; the gradient test
    (``settings.grad_test``) on every member's bounded gradient; every member's solve at
    zero damping (under its own rows of ``rows``, ``owners`` the static
    ``where.owners``), its predicted decrease and stationarity number, the stationarity
    test (against the yardstick, raised to ``level`` times its full step's norm once the
    floor is measured) and the accuracy test; then, while a member is active and
    ``before.stepping``, every active member's step (``_step``).
    """
    level, count = settings.level, settings.count
    x = before.point
    member_of = where.owners
    gradient, scale, subproblem = coordinates(system, x_scale, x, before.scale, bounds)
    addition = jnp.zeros_like(scale) if subproblem.addition is None else subproblem.addition
    active, codes = before.active, before.codes

    def finish(done, code, active, codes):
        return active & ~done, jnp.where(done, code, codes)

    if settings.grad_test:
        norms = _member_norms(_bounded_gradient(gradient, subproblem, bounds), member_of, count)
        active, codes = finish(
            active & (norms <= before.thresholds), GRADIENT_TOLERANCE, active, codes
        )
    own = None if rows is None else inequality.member_row_set(rows, system, x, level, owners)
    cscale = subproblem.scale
    if own is None:
        damping = jnp.zeros((count,), dtype=cscale.dtype)
        zero = first = _inner_solve(_scaled(system, cscale, addition), damping, level=level)
        zero_solves = jnp.asarray(0)
    else:
        zero, first, zero_solves = inequality.member_zero(own, subproblem, x, level, owners)
    z, q_squared = zero
    decreases = _member_predicted(system, z / cscale, z, addition, member_of, count, level=level)
    broken = _member_sums((~jnp.isfinite(z)).astype(jnp.int32), member_of, count) > 0
    decreases = jnp.where(broken | ~jnp.isfinite(q_squared), jnp.nan, decreases)
    relative_to = cost_scale(before.costs, settings.cost_tol, settings.cost_atol)
    stationarity = decreases / relative_to
    full_steps = _member_norms(z, member_of, count) / relative_to
    limits = jnp.where(
        jnp.isnan(before.level), before.limit, jnp.fmax(before.limit, before.level * full_steps)
    )
    active, codes = finish(active & _within(stationarity, limits), STATIONARITY, active, codes)
    active, codes = finish(active & _within(decreases, before.accuracy), ACCURACY, active, codes)
    stepped = jnp.any(active) & before.stepping

    def make() -> tuple:
        trial, step_norms, scaled_norms, predicted, no_step, solves = _step(
            system,
            where,
            x,
            gradient,
            subproblem,
            addition,
            before.radii,
            active,
            first,
            own,
            bounds,
            settings,
            rows,
            owners,
        )
        dtype = x.dtype
        return (
            trial.astype(dtype),
            step_norms.astype(dtype),
            scaled_norms.astype(dtype),
            predicted.astype(dtype),
            no_step,
            jnp.asarray(solves, dtype=jnp.int32),
        )

    def skip() -> tuple:
        none = jnp.zeros((count,), dtype=x.dtype)
        return x, none, none, none, jnp.zeros((count,), dtype=bool), jnp.asarray(0, jnp.int32)

    trial, step_norms, scaled_norms, predicted, no_step, solves = jax.lax.cond(stepped, make, skip)
    active, codes = finish(no_step, NO_FINITE_STEP, active, codes)
    return MemberProposal(
        scale,
        active,
        codes,
        stationarity,
        stepped,
        trial,
        step_norms,
        scaled_norms,
        predicted,
        zero_solves,
        solves,
    )


propose_program = jax.jit(propose, static_argnames=("settings",))


class MemberOutcome(NamedTuple):
    """What the host learned at the members' combined trial, and the proposal it judges.

    ``passing`` are the members whose trial the predicate passed and the request
    covered, ``failed`` those whose lanes failed there; ``evaluated`` whether the request
    was made; ``costs`` every member's cost at the point (after the re-solves of the
    lanes whose rung rose) and ``trial_costs`` at the trial (``inf`` where not known);
    ``noise`` the evaluation noise the outer trials measured.
    """

    active: ArrayLike
    passing: ArrayLike
    failed: ArrayLike
    evaluated: ArrayLike
    costs: ArrayLike
    trial_costs: ArrayLike
    predicted: ArrayLike
    scaled_norms: ArrayLike
    step_norms: ArrayLike
    point: ArrayLike
    trial: ArrayLike
    noise: ArrayLike


class MemberState(NamedTuple):
    """The members' state the polish carries from one iteration to the next.

    ``noise_runs`` are every member's noise-dominated rejected trials in a row and
    ``accepted_radii`` the radius of its last accepted trial in this polish (``nan``
    before the first).
    """

    radii: ArrayLike
    locks: ArrayLike
    noise_runs: ArrayLike
    accepted_radii: ArrayLike
    codes: ArrayLike


def judge(where: Layout, outcome: MemberOutcome, state: MemberState, *, settings: PolishSettings):
    """Every member's verdict on its trial, as the joint attempt's.

    Its gain ratio (relaxed by the outer trials' noise when the inner
    ``noise_relaxed_ratio`` is set), acceptance, radius and growth lock; its run of
    noise-dominated rejected trials (a trial whose cost was computed extends or ends it,
    a failed one ends it, one the predicate rejected leaves it); its step and cost tests,
    which read its trial whenever the trial's cost was computed, accepted or rejected;
    and its radius floor. Returns ``(accepted, point, state, active, floored)``: the
    members' point after their accepted trials.
    """
    count = settings.count
    member_of = where.owners
    o = outcome
    relative_to = cost_scale(o.costs, settings.cost_tol, settings.cost_atol)
    gains = o.evaluated & o.passing & ~o.failed & (o.predicted > 0.0)
    ratio = _gain_ratio(
        o.costs - o.trial_costs,
        o.predicted,
        noise=o.noise,
        relative_to=relative_to,
        relaxed=settings.relaxed,
    )
    rho = jnp.where(gains, ratio, -jnp.inf)
    active = o.active
    accepted = active & (rho >= settings.eta)
    computed = o.evaluated & o.passing & ~o.failed
    radii = jnp.where(
        active,
        _next_radius(
            state.radii,
            rho,
            o.scaled_norms,
            lock=state.locks,
            radius_max=settings.radius_max,
            computed=computed,
        ),
        state.radii,
    )
    locks = jnp.where(
        active & accepted,
        jnp.where(state.locks > 0, state.locks - 1, state.locks),
        jnp.where(active & o.failed, 2, state.locks),
    )
    owned = member_of >= 0
    point = jnp.where(owned & accepted[jnp.where(owned, member_of, 0)], o.trial, o.point)
    computed = computed & jnp.isfinite(o.trial_costs)
    noisy = _noise_dominated(
        evaluated=computed,
        accepted=accepted,
        step_radius=state.radii,
        accepted_radius=state.accepted_radii,
        cost=o.costs,
        trial_cost=o.trial_costs,
        predicted=o.predicted,
        relative_to=relative_to,
        limit=settings.noise_limit,
    )
    noise_runs = jnp.where(
        active & computed,
        jnp.where(noisy, state.noise_runs + 1, 0),
        jnp.where(active & o.failed, 0, state.noise_runs),
    )
    accepted_radii = jnp.where(active & accepted, state.radii, state.accepted_radii)
    small_step = step_small(
        o.step_norms, _member_norms(point, member_of, count), settings.step_tol, tested=computed
    )
    small_cost = cost_small(
        accepted=accepted,
        evaluated=computed,
        cost_change=jnp.abs(o.trial_costs - o.costs),
        predicted=o.predicted,
        threshold=cost_threshold(o.costs, settings.cost_tol, settings.cost_atol),
    )
    noise_run = noise_runs >= _NOISE_RUN
    codes = jnp.where(
        active & small_step,
        STEP_TOLERANCE,
        jnp.where(
            active & small_cost,
            COST_TOLERANCE,
            jnp.where(active & noise_run, EVALUATION_NOISE, state.codes),
        ),
    )
    active = active & ~(small_step | small_cost | noise_run)
    floored = active & (radii <= settings.tiny)
    codes = jnp.where(floored, RADIUS, codes)
    state = MemberState(radii, locks, noise_runs, accepted_radii, codes)
    return accepted, point, state, active & ~floored, floored


judge_program = jax.jit(judge, static_argnames=("settings",))


__all__ = [
    "Layout",
    "MemberBefore",
    "MemberOutcome",
    "MemberProposal",
    "MemberState",
    "PolishSettings",
    "judge",
    "judge_program",
    "layout",
    "propose",
    "propose_program",
]
