"""The bound handling of the trust-region step, after Branch, Coleman and Li.

These functions reproduce the handling of box bounds in SciPy's
``least_squares`` with ``method="trf"`` (``scipy/optimize/_lsq/common.py`` and
``trf.py``), under SciPy's names and with SciPy's semantics, elementwise, with
infinite sides allowed. Each is written for arrays: on NumPy values it computes on
the host, with SciPy's early return and errors; on traced values it computes in the
program that traces it, every branch a selection (``_loop_judge._namespace``). One difference: the quadratic model enters as a
callable ``quadratic_form(v) = v^T H_h v`` of a vector ``v`` in the scaled
coordinates, in place of a dense Jacobian ``J_h`` and a diagonal ``diag_h``
(``H_h = J_h^T J_h + diag(diag_h)``), because the model of a parameter
hierarchy is evaluated along its tree and never formed as a dense matrix.

At a point ``x`` strictly inside the box ``lb <= x <= ub`` with gradient
``g``, the Coleman-Li vector ``v`` holds, for every coordinate whose descent
direction ``-g`` points at a finite side, the distance to that side, and 1
otherwise. The trust-region subproblem is solved in the coordinates ``p_h`` of
``s = d * p_h`` with ``d = sqrt(v_hat) / scale`` (``v_hat`` is ``v`` times the
scale where ``v`` is a distance), and the diagonal ``g * dv / scale`` is added
to the scaled matrix. So the trust region narrows along a coordinate as it
nears the side its gradient pushes toward. When the step leaves the box,
``select_step`` takes the best of three candidates by the model's value: the
step shortened to stay inside, the step reflected at the sides it crosses, and
the Cauchy step along ``-g_h``. Iterates are kept strictly inside the box by
``make_strictly_feasible``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Real
from typing import NamedTuple

import numpy as np
from jaxtyping import ArrayLike

from diff_soup.optimization.solvers._loop_judge import _larger, _namespace, _smaller


def CL_scaling_vector(
    x: ArrayLike | Sequence[Real],
    g: ArrayLike | Sequence[Real],
    lb: ArrayLike | Sequence[Real],
    ub: ArrayLike | Sequence[Real],
) -> tuple:
    """The Coleman-Li scaling vector ``v`` and its derivative ``dv`` at ``x``.

    ``v[i]`` is ``ub[i] - x[i]`` where ``g[i] < 0`` and ``ub[i]`` is finite,
    ``x[i] - lb[i]`` where ``g[i] > 0`` and ``lb[i]`` is finite, and 1
    otherwise; ``dv[i]`` is -1, 1 or 0 accordingly.

    Args:
        x: ``(n,)``, a point inside the box.
        g: ``(n,)``, the gradient at ``x``.
        lb: ``(n,)``, the lower sides, ``-inf`` where there is none.
        ub: ``(n,)``, the upper sides, ``inf`` where there is none.

    Returns:
        ``(v, dv)``, both of the shape of ``x``.
    """
    xp = _namespace(x, g, lb, ub)
    x, g, lb, ub = (xp.asarray(value) for value in (x, g, lb, ub))
    upper = (g < 0) & xp.isfinite(ub)
    lower = (g > 0) & xp.isfinite(lb)
    with np.errstate(invalid="ignore"):
        v = xp.where(lower, x - lb, xp.where(upper, ub - x, xp.ones_like(x)))
    dv = xp.where(lower, 1.0, xp.where(upper, -1.0, 0.0)).astype(x.dtype)
    return v, dv


def find_active_constraints(
    x: ArrayLike | Sequence[Real],
    lb: ArrayLike | Sequence[Real],
    ub: ArrayLike | Sequence[Real],
    rtol: Real = 1e-10,
) -> ArrayLike:
    """Which side of the box is active at ``x``, per coordinate.

    A side is active when ``x`` is within ``rtol * max(1, |side|)`` of it and
    no farther from it than from the other side; with ``rtol = 0`` when ``x``
    is on it or beyond it.

    Args:
        x: ``(n,)``, a point.
        lb: ``(n,)``, the lower sides.
        ub: ``(n,)``, the upper sides.
        rtol: The relative distance within which a side is active.

    Returns:
        ``(n,)`` integers: 0 where no side is active, -1 where the lower side
        is, 1 where the upper side is.
    """
    xp = _namespace(x, lb, ub)
    x, lb, ub = (xp.asarray(value) for value in (x, lb, ub))
    if rtol == 0:
        return xp.where(x >= ub, 1, xp.where(x <= lb, -1, 0))
    with np.errstate(all="ignore"):
        lower_dist = x - lb
        upper_dist = ub - x
        lower_threshold = rtol * xp.maximum(1, xp.abs(lb))
        upper_threshold = rtol * xp.maximum(1, xp.abs(ub))
        lower_active = xp.isfinite(lb) & (lower_dist <= xp.minimum(upper_dist, lower_threshold))
        upper_active = xp.isfinite(ub) & (upper_dist <= xp.minimum(lower_dist, upper_threshold))
    return xp.where(upper_active, 1, xp.where(lower_active, -1, 0))


def make_strictly_feasible(
    x: ArrayLike | Sequence[Real],
    lb: ArrayLike | Sequence[Real],
    ub: ArrayLike | Sequence[Real],
    rstep: Real = 0,
) -> ArrayLike:
    """``x`` moved strictly inside the box.

    With ``rstep = 0`` a coordinate on a side or beyond it is moved to the next
    float inside that side; otherwise a coordinate within ``rstep * max(1,
    |side|)`` of a side is moved to that distance from it. A coordinate whose
    sides are too close for either is put at their midpoint.

    Args:
        x: ``(n,)``, a point.
        lb: ``(n,)``, the lower sides.
        ub: ``(n,)``, the upper sides.
        rstep: The relative distance kept from a side; 0 for the next float.

    Returns:
        ``(n,)``, a new array.
    """
    xp = _namespace(x, lb, ub)
    x, lb, ub = (xp.asarray(value) for value in (x, lb, ub))
    active = find_active_constraints(x, lb, ub, rstep)
    if rstep == 0:
        lower, upper = xp.nextafter(lb, ub), xp.nextafter(ub, lb)
    else:
        lower = lb + rstep * xp.maximum(1, xp.abs(lb))
        upper = ub - rstep * xp.maximum(1, xp.abs(ub))
    x_new = xp.where(active == 1, upper, xp.where(active == -1, lower, x)).astype(x.dtype)
    tight_bounds = (x_new < lb) | (x_new > ub)
    with np.errstate(all="ignore"):
        return xp.where(tight_bounds, 0.5 * (lb + ub), x_new).astype(x.dtype)


def in_bounds(
    x: ArrayLike | Sequence[Real], lb: ArrayLike | Sequence[Real], ub: ArrayLike | Sequence[Real]
) -> ArrayLike:
    """Whether ``lb <= x <= ub`` holds in every coordinate."""
    xp = _namespace(x, lb, ub)
    inside = xp.all((xp.asarray(x) >= lb) & (xp.asarray(x) <= ub))
    return bool(inside) if xp is np else inside


def step_size_to_bound(
    x: ArrayLike | Sequence[Real],
    s: ArrayLike | Sequence[Real],
    lb: ArrayLike | Sequence[Real],
    ub: ArrayLike | Sequence[Real],
) -> tuple:
    """The smallest ``t >= 0`` at which ``x + t * s`` reaches a side of the box.

    Args:
        x: ``(n,)``, a point inside the box.
        s: ``(n,)``, a direction.
        lb: ``(n,)``, the lower sides.
        ub: ``(n,)``, the upper sides.

    Returns:
        ``(t, hits)``: ``t`` (``inf`` when no side is reached) and ``(n,)``
        integers, -1 where the lower side is reached at ``t``, 1 where the
        upper side is, 0 elsewhere.
    """
    xp = _namespace(x, s, lb, ub)
    x, s, lb, ub = (xp.asarray(value) for value in (x, s, lb, ub))
    moving = s != 0
    divisor = xp.where(moving, s, 1)
    with np.errstate(all="ignore"):
        steps = xp.where(moving, xp.maximum((lb - x) / divisor, (ub - x) / divisor), xp.inf)
    min_step = xp.min(steps) if steps.size else xp.asarray(xp.inf, dtype=x.dtype)
    return min_step, xp.equal(steps, min_step) * xp.sign(s).astype(int)


def intersect_trust_region(
    x: ArrayLike | Sequence[Real], s: ArrayLike | Sequence[Real], Delta: ArrayLike
) -> tuple:
    """The two roots ``t`` of ``||x + t * s|| = Delta``, the negative one first.

    Raises:
        ValueError: If ``s`` is zero or ``x`` is outside the trust region (on
            the host; a traced call gives ``nan`` there).
    """
    xp = _namespace(x, s, Delta)
    x, s = xp.asarray(x), xp.asarray(s)
    a = xp.dot(s, s)
    b = xp.dot(x, s)
    c = xp.dot(x, x) - Delta**2
    if xp is np:
        if a == 0:
            raise ValueError("`s` is zero.")
        if c > 0:
            raise ValueError("`x` is not within the trust region.")
    with np.errstate(all="ignore"):
        d = xp.sqrt(b * b - a * c)
        q = -(b + xp.copysign(d, b))
        t1 = q / a
        t2 = c / q
    if xp is np:
        return (t1, t2) if t1 < t2 else (t2, t1)
    return xp.where(t1 < t2, t1, t2), xp.where(t1 < t2, t2, t1)


def build_quadratic_1d(
    quadratic_form: Callable[..., ArrayLike],
    g: ArrayLike | Sequence[Real],
    s: ArrayLike | Sequence[Real],
    s0: ArrayLike | Sequence[Real] | None = None,
) -> tuple:
    """The model ``0.5 u^T H u + g^T u`` along the line ``u = s0 + t * s``.

    Args:
        quadratic_form: ``quadratic_form(v) = v^T H v``.
        g: ``(n,)``, the gradient of the model.
        s: ``(n,)``, the direction of the line.
        s0: ``(n,)``, the line's point at ``t = 0``; ``None`` for the origin.

    Returns:
        ``(a, b)`` of ``a t^2 + b t`` without ``s0``, ``(a, b, c)`` of
        ``a t^2 + b t + c`` with it.
    """
    xp = _namespace(g, s, s0)
    s = xp.asarray(s)
    curvature = xp.asarray(quadratic_form(s))
    a = 0.5 * curvature
    b = xp.dot(g, s)
    if s0 is None:
        return a, b
    s0 = xp.asarray(s0)
    at_start = xp.asarray(quadratic_form(s0))
    b = b + 0.5 * (xp.asarray(quadratic_form(s0 + s)) - at_start - curvature)
    c = 0.5 * at_start + xp.dot(g, s0)
    return a, b, c


def minimize_quadratic_1d(
    a: ArrayLike, b: ArrayLike, lb: ArrayLike, ub: ArrayLike, c: ArrayLike = 0
) -> tuple:
    """The minimum of ``a t^2 + b t + c`` on ``lb <= t <= ub`` (finite ends).

    Returns:
        ``(t, y)``, the minimizer and the minimum.
    """
    xp = _namespace(a, b, lb, ub, c)
    with np.errstate(all="ignore"):
        extremum = -0.5 * b / xp.where(a != 0, a, 1)
        inside = (a != 0) & (lb < extremum) & (extremum < ub)
        t = xp.stack([xp.asarray(lb, dtype=float), xp.asarray(ub, dtype=float), extremum])
        y = t * (a * t + b) + c
    y = xp.where(xp.asarray([True, True, False]) | inside, y, xp.inf)
    min_index = xp.argmin(y)
    return t[min_index], y[min_index]


def evaluate_quadratic(
    quadratic_form: Callable[..., ArrayLike],
    g: ArrayLike | Sequence[Real],
    s: ArrayLike | Sequence[Real],
) -> ArrayLike:
    """The model ``0.5 s^T H s + g^T s`` at the step ``s``."""
    xp = _namespace(g, s)
    s = xp.asarray(s)
    return 0.5 * xp.asarray(quadratic_form(s)) + xp.dot(s, g)


def select_step(
    x: ArrayLike | Sequence[Real],
    quadratic_form: Callable[..., ArrayLike],
    g_h: ArrayLike | Sequence[Real],
    p: ArrayLike | Sequence[Real],
    p_h: ArrayLike | Sequence[Real],
    d: ArrayLike | Sequence[Real],
    Delta: ArrayLike,
    lb: ArrayLike | Sequence[Real],
    ub: ArrayLike | Sequence[Real],
    theta: ArrayLike,
) -> tuple:
    """The trial step of the trust-region subproblem's solution under the box.

    When ``x + p`` is in the box, ``p`` is the step. Otherwise three
    candidates are formed: ``p`` shortened to ``theta`` times the distance to
    the first side it reaches; ``p`` reflected at the sides it crosses (the
    components of ``p_h`` that cross flipped), its length the minimizer of the
    model between the box, the trust region and a strict-feasibility margin;
    and the Cauchy step along ``-g_h`` within the trust region and the box. The
    candidate of the lowest model value is the step.

    Args:
        x: ``(n,)``, the current point.
        quadratic_form: ``quadratic_form(v) = v^T H_h v`` for a scaled vector
            ``v``, ``H_h`` the scaled model matrix with its diagonal addition.
        g_h: ``(n,)``, the scaled gradient ``d * g``.
        p: ``(n,)``, the subproblem's step ``d * p_h``.
        p_h: ``(n,)``, the subproblem's step in the scaled coordinates.
        d: ``(n,)``, the scaling ``s = d * p_h``.
        Delta: The trust-region radius, at least ``||p_h||``.
        lb: ``(n,)``, the lower sides.
        ub: ``(n,)``, the upper sides.
        theta: The fraction of the distance to a side a step may take.

    Returns:
        ``(step, step_h, predicted_reduction)``: the step, its scaled
        coordinates, and the decrease the model predicts for it.

    Raises:
        ValueError: If ``||p_h||`` is larger than ``Delta`` (on the host).
    """
    with np.errstate(all="ignore"):
        return _candidates(x, quadratic_form, g_h, p, p_h, d, Delta, lb, ub, theta)


def _candidates(x, quadratic_form, g_h, p, p_h, d, Delta, lb, ub, theta) -> tuple:
    """``select_step``'s statements, in SciPy's order; on traced values every branch is taken."""
    xp = _namespace(x, g_h, p, p_h, d, Delta, theta)
    x, g_h, d, p, p_h = (xp.asarray(value) for value in (x, g_h, d, p, p_h))
    inside = in_bounds(x + p, lb, ub)
    full_value = evaluate_quadratic(quadratic_form, g_h, p_h)
    if xp is np and inside:
        return p.copy(), p_h.copy(), -full_value

    p_stride, hits = step_size_to_bound(x, p, lb, ub)

    # The reflected direction.
    r_h = xp.where(hits != 0, -p_h, p_h)
    r = d * r_h

    # The step restricted to reach the side.
    p_bound = p * p_stride
    p_h_bound = p_h * p_stride
    x_on_bound = x + p_bound

    # The reflected direction crosses the box's or the trust region's boundary first.
    _, to_tr = intersect_trust_region(p_h_bound, r_h, Delta)
    to_bound, _ = step_size_to_bound(x_on_bound, r, lb, ub)

    # The range of the step along the reflected direction, with a margin from the box.
    r_stride = _smaller(to_bound, to_tr)
    positive = r_stride > 0
    r_stride_l = xp.where(positive, (1 - theta) * p_stride / r_stride, 0)
    r_stride_u = xp.where(positive, xp.where(r_stride == to_bound, theta * to_bound, to_tr), -1)
    reflected = r_stride_l <= r_stride_u
    a, b, c = build_quadratic_1d(quadratic_form, g_h, r_h, s0=p_h_bound)
    t, value = minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c=c)
    r_h = xp.where(reflected, r_h * t + p_h_bound, r_h)
    r = r_h * d
    r_value = xp.where(reflected, value, xp.inf)

    # The shortened step, strictly inside.
    p_short = p_bound * theta
    p_h_short = p_h_bound * theta
    p_value = evaluate_quadratic(quadratic_form, g_h, p_h_short)

    ag_h = -g_h
    ag = d * ag_h
    to_tr = Delta / xp.linalg.norm(ag_h)
    to_bound, _ = step_size_to_bound(x, ag, lb, ub)
    ag_stride = xp.where(to_bound < to_tr, theta * to_bound, to_tr)
    a, b = build_quadratic_1d(quadratic_form, g_h, ag_h)
    ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride)
    ag_h = ag_h * ag_stride
    ag = ag * ag_stride

    take_p = (p_value < r_value) & (p_value < ag_value)
    take_r = ~take_p & (r_value < p_value) & (r_value < ag_value)
    step = xp.where(inside, p, xp.where(take_p, p_short, xp.where(take_r, r, ag)))
    step_h = xp.where(inside, p_h, xp.where(take_p, p_h_short, xp.where(take_r, r_h, ag_h)))
    value = xp.where(
        inside, full_value, xp.where(take_p, p_value, xp.where(take_r, r_value, ag_value))
    )
    return step, step_h, -value


def shortened_step(
    x: ArrayLike | Sequence[Real],
    quadratic_form: Callable[..., ArrayLike],
    g_h: ArrayLike | Sequence[Real],
    p: ArrayLike | Sequence[Real],
    p_h: ArrayLike | Sequence[Real],
    lb: ArrayLike | Sequence[Real],
    ub: ArrayLike | Sequence[Real],
    theta: ArrayLike,
) -> tuple:
    """``select_step``'s first candidate alone: ``p``, or ``p`` shortened to stay inside.

    The step that keeps the subproblem's equality rows: a multiple of ``p``.

    Returns:
        ``(step, step_h, predicted_reduction)`` as ``select_step`` returns them.
    """
    xp = _namespace(x, g_h, p, p_h, theta)
    x, p, p_h = xp.asarray(x), xp.asarray(p), xp.asarray(p_h)
    with np.errstate(all="ignore"):
        inside = in_bounds(x + p, lb, ub)
        p_stride, _ = step_size_to_bound(x, p, lb, ub)
        p = xp.where(inside, p, (p * p_stride) * theta)
        p_h = xp.where(inside, p_h, (p_h * p_stride) * theta)
        return p, p_h, -evaluate_quadratic(quadratic_form, g_h, p_h)


def step_back_ratio(g: ArrayLike | Sequence[Real], v: ArrayLike | Sequence[Real]) -> ArrayLike:
    """``theta = max(0.995, 1 - ||g * v||_inf)``: how close to a side a step may go."""
    xp = _namespace(g, v)
    product = xp.asarray(g) * xp.asarray(v)
    largest = xp.max(xp.abs(product)) if product.size else 0.0
    if xp is np:
        return max(0.995, 1.0 - float(largest))
    return _larger(0.995, 1.0 - largest)


class Scaling(NamedTuple):
    """The trust-region subproblem's scaled coordinates at a point.

    Attributes:
        v: ``(n,)``, the Coleman-Li vector.
        dv: ``(n,)``, its derivative, -1, 0 or 1.
        scale: ``(n,)``, the scale of the coordinates: ``p_h = scale * s``,
            ``1 / d``.
        addition: ``(n,)``, the diagonal ``g * dv / S`` added to the scaled
            matrix; zero where ``dv`` is 0.
        theta: ``max(0.995, 1 - ||g * v||_inf)``.
    """

    v: ArrayLike
    dv: ArrayLike
    scale: ArrayLike
    addition: ArrayLike | None
    theta: ArrayLike


def coleman_li_scaling(
    x: ArrayLike | Sequence[Real],
    g: ArrayLike | Sequence[Real],
    lb: ArrayLike | Sequence[Real],
    ub: ArrayLike | Sequence[Real],
    scale: ArrayLike | Sequence[Real],
) -> Scaling:
    """The scaled coordinates of the trust-region subproblem at ``x``.

    ``scale`` is the solver's scale ``S`` of the variables (the step's scaled
    norm without bounds is ``||S * s||``); it plays the part of SciPy's
    ``scale_inv``. With ``v_hat = v * S`` where ``dv != 0`` and ``v_hat = 1``
    elsewhere, the step is ``s = d * p_h``, ``d = sqrt(v_hat) / S``, and
    ``g * dv / S`` is added to the diagonal of the scaled matrix. Where no side
    is finite the scale is ``S`` and the addition zero.

    Args:
        x: ``(n,)``, the point, strictly inside the box.
        g: ``(n,)``, the gradient at ``x``.
        lb: ``(n,)``, the lower sides.
        ub: ``(n,)``, the upper sides.
        scale: ``(n,)``, positive.

    Returns:
        The [Scaling][diff_soup.optimization.inference.hierarchical._bounds.Scaling].
    """
    xp = _namespace(x, g, scale)
    scale = xp.asarray(scale)
    g = xp.asarray(g)
    v, dv = CL_scaling_vector(x, g, lb, ub)
    theta = step_back_ratio(g, v)
    bounded = dv != 0
    v_hat = xp.where(bounded, v * scale, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        coordinates = scale / xp.sqrt(v_hat)
        addition = xp.where(bounded, g * dv / scale, 0.0).astype(scale.dtype)
    return Scaling(v=v, dv=dv, scale=coordinates, addition=addition, theta=theta)


def active_bounds(
    x: ArrayLike | Sequence[Real],
    bounds: tuple[ArrayLike | Sequence[Real], ArrayLike | Sequence[Real]] | None,
    rtol: Real = 1e-10,
) -> tuple:
    """The active sides at ``x`` as ``((index, -1 | 1), ...)``; empty without bounds."""
    if bounds is None:
        return ()
    active = find_active_constraints(np.asarray(x), bounds[0], bounds[1], rtol=rtol)
    return tuple((int(index), int(active[index])) for index in np.flatnonzero(active))


__all__ = [
    "CL_scaling_vector",
    "Scaling",
    "active_bounds",
    "build_quadratic_1d",
    "coleman_li_scaling",
    "evaluate_quadratic",
    "find_active_constraints",
    "in_bounds",
    "intersect_trust_region",
    "make_strictly_feasible",
    "minimize_quadratic_1d",
    "select_step",
    "shortened_step",
    "step_back_ratio",
    "step_size_to_bound",
]
