"""The trust-region step of a lowered study: its subproblem, its trial, its noise and gradient floors."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from diff_soup.optimization.inference.engine._layout import lane_layout
from diff_soup.optimization.inference.fitting._trial import trial_is_valid
from diff_soup.optimization.inference.fitting.scales import ConstantScale, HessianScale
from diff_soup.optimization.inference.hierarchical import _bounds as box
from diff_soup.optimization.inference.hierarchical import _level_elimination as elimination
from diff_soup.optimization.inference.hierarchical._lane_route import _lane_positions
from diff_soup.optimization.solvers._loop import (
    _floor_radius,
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
