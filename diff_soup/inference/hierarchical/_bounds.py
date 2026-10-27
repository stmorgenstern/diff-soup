"""The bound handling of the trust-region step, after Branch, Coleman and Li.

These host-side functions reproduce the handling of box bounds in SciPy's
``least_squares`` with ``method="trf"`` (``scipy/optimize/_lsq/common.py`` and
``trf.py``), under SciPy's names and with SciPy's semantics, elementwise, with
infinite sides allowed. One difference: the quadratic model enters as a
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

from math import copysign
from typing import Any, NamedTuple

import numpy as np


def CL_scaling_vector(x: Any, g: Any, lb: Any, ub: Any) -> tuple:
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
    x, g, lb, ub = (np.asarray(value) for value in (x, g, lb, ub))
    v = np.ones_like(x)
    dv = np.zeros_like(x)
    mask = (g < 0) & np.isfinite(ub)
    v[mask] = ub[mask] - x[mask]
    dv[mask] = -1
    mask = (g > 0) & np.isfinite(lb)
    v[mask] = x[mask] - lb[mask]
    dv[mask] = 1
    return v, dv


def find_active_constraints(x: Any, lb: Any, ub: Any, rtol: Any = 1e-10) -> np.ndarray:
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
    x, lb, ub = (np.asarray(value) for value in (x, lb, ub))
    active = np.zeros_like(x, dtype=int)
    if rtol == 0:
        active[x <= lb] = -1
        active[x >= ub] = 1
        return active
    with np.errstate(all="ignore"):
        lower_dist = x - lb
        upper_dist = ub - x
        lower_threshold = rtol * np.maximum(1, np.abs(lb))
        upper_threshold = rtol * np.maximum(1, np.abs(ub))
        lower_active = np.isfinite(lb) & (lower_dist <= np.minimum(upper_dist, lower_threshold))
        upper_active = np.isfinite(ub) & (upper_dist <= np.minimum(lower_dist, upper_threshold))
    active[lower_active] = -1
    active[upper_active] = 1
    return active


def make_strictly_feasible(x: Any, lb: Any, ub: Any, rstep: Any = 0) -> np.ndarray:
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
    x, lb, ub = (np.asarray(value) for value in (x, lb, ub))
    x_new = x.copy()
    active = find_active_constraints(x, lb, ub, rstep)
    lower_mask = np.equal(active, -1)
    upper_mask = np.equal(active, 1)
    if rstep == 0:
        x_new[lower_mask] = np.nextafter(lb[lower_mask], ub[lower_mask])
        x_new[upper_mask] = np.nextafter(ub[upper_mask], lb[upper_mask])
    else:
        x_new[lower_mask] = lb[lower_mask] + rstep * np.maximum(1, np.abs(lb[lower_mask]))
        x_new[upper_mask] = ub[upper_mask] - rstep * np.maximum(1, np.abs(ub[upper_mask]))
    tight_bounds = (x_new < lb) | (x_new > ub)
    with np.errstate(all="ignore"):
        x_new[tight_bounds] = 0.5 * (lb[tight_bounds] + ub[tight_bounds])
    return x_new


def in_bounds(x: Any, lb: Any, ub: Any) -> bool:
    """Whether ``lb <= x <= ub`` holds in every coordinate."""
    return bool(np.all((np.asarray(x) >= lb) & (np.asarray(x) <= ub)))


def step_size_to_bound(x: Any, s: Any, lb: Any, ub: Any) -> tuple:
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
    x, s, lb, ub = (np.asarray(value) for value in (x, s, lb, ub))
    non_zero = np.nonzero(s)
    s_non_zero = s[non_zero]
    steps = np.empty_like(x)
    steps.fill(np.inf)
    with np.errstate(over="ignore"):
        steps[non_zero] = np.maximum(
            (lb - x)[non_zero] / s_non_zero, (ub - x)[non_zero] / s_non_zero
        )
    min_step = np.min(steps)
    return min_step, np.equal(steps, min_step) * np.sign(s).astype(int)


def intersect_trust_region(x: Any, s: Any, Delta: Any) -> tuple:
    """The two roots ``t`` of ``||x + t * s|| = Delta``, the negative one first.

    Raises:
        ValueError: If ``s`` is zero or ``x`` is outside the trust region.
    """
    x, s = np.asarray(x), np.asarray(s)
    a = np.dot(s, s)
    if a == 0:
        raise ValueError("`s` is zero.")
    b = np.dot(x, s)
    c = np.dot(x, x) - Delta**2
    if c > 0:
        raise ValueError("`x` is not within the trust region.")
    d = np.sqrt(b * b - a * c)
    q = -(b + copysign(d, b))
    t1 = q / a
    t2 = c / q
    if t1 < t2:
        return t1, t2
    return t2, t1


def build_quadratic_1d(quadratic_form: Any, g: Any, s: Any, s0: Any = None) -> tuple:
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
    s = np.asarray(s)
    curvature = float(quadratic_form(s))
    a = 0.5 * curvature
    b = np.dot(g, s)
    if s0 is None:
        return a, b
    s0 = np.asarray(s0)
    at_start = float(quadratic_form(s0))
    b += 0.5 * (float(quadratic_form(s0 + s)) - at_start - curvature)
    c = 0.5 * at_start + np.dot(g, s0)
    return a, b, c


def minimize_quadratic_1d(a: Any, b: Any, lb: Any, ub: Any, c: Any = 0) -> tuple:
    """The minimum of ``a t^2 + b t + c`` on ``lb <= t <= ub`` (finite ends).

    Returns:
        ``(t, y)``, the minimizer and the minimum.
    """
    t = [lb, ub]
    if a != 0:
        extremum = -0.5 * b / a
        if lb < extremum < ub:
            t.append(extremum)
    t = np.asarray(t)
    y = t * (a * t + b) + c
    min_index = np.argmin(y)
    return t[min_index], y[min_index]


def evaluate_quadratic(quadratic_form: Any, g: Any, s: Any) -> Any:
    """The model ``0.5 s^T H s + g^T s`` at the step ``s``."""
    s = np.asarray(s)
    return 0.5 * float(quadratic_form(s)) + float(np.dot(s, g))


def select_step(
    x: Any,
    quadratic_form: Any,
    g_h: Any,
    p: Any,
    p_h: Any,
    d: Any,
    Delta: Any,
    lb: Any,
    ub: Any,
    theta: Any,
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
        ValueError: If ``||p_h||`` is larger than ``Delta``.
    """
    with np.errstate(all="ignore"):
        return _candidates(x, quadratic_form, g_h, p, p_h, d, Delta, lb, ub, theta)


def _candidates(x, quadratic_form, g_h, p, p_h, d, Delta, lb, ub, theta) -> tuple:
    """``select_step``'s statements, in SciPy's order."""
    x, g_h, d = np.asarray(x), np.asarray(g_h), np.asarray(d)
    p = np.array(p, copy=True)
    p_h = np.array(p_h, copy=True)
    if in_bounds(x + p, lb, ub):
        p_value = evaluate_quadratic(quadratic_form, g_h, p_h)
        return p, p_h, -p_value

    p_stride, hits = step_size_to_bound(x, p, lb, ub)

    # The reflected direction.
    r_h = np.copy(p_h)
    r_h[hits.astype(bool)] *= -1
    r = d * r_h

    # The step restricted to reach the side.
    p *= p_stride
    p_h *= p_stride
    x_on_bound = x + p

    # The reflected direction crosses the box's or the trust region's boundary first.
    _, to_tr = intersect_trust_region(p_h, r_h, Delta)
    to_bound, _ = step_size_to_bound(x_on_bound, r, lb, ub)

    # The range of the step along the reflected direction, with a margin from the box.
    r_stride = min(to_bound, to_tr)
    if r_stride > 0:
        r_stride_l = (1 - theta) * p_stride / r_stride
        if r_stride == to_bound:
            r_stride_u = theta * to_bound
        else:
            r_stride_u = to_tr
    else:
        r_stride_l = 0
        r_stride_u = -1

    if r_stride_l <= r_stride_u:
        a, b, c = build_quadratic_1d(quadratic_form, g_h, r_h, s0=p_h)
        r_stride, r_value = minimize_quadratic_1d(a, b, r_stride_l, r_stride_u, c=c)
        r_h *= r_stride
        r_h += p_h
        r = r_h * d
    else:
        r_value = np.inf

    # The shortened step, strictly inside.
    p *= theta
    p_h *= theta
    p_value = evaluate_quadratic(quadratic_form, g_h, p_h)

    ag_h = -g_h
    ag = d * ag_h
    to_tr = Delta / np.linalg.norm(ag_h)
    to_bound, _ = step_size_to_bound(x, ag, lb, ub)
    if to_bound < to_tr:
        ag_stride = theta * to_bound
    else:
        ag_stride = to_tr
    a, b = build_quadratic_1d(quadratic_form, g_h, ag_h)
    ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride)
    ag_h = ag_h * ag_stride
    ag = ag * ag_stride

    if p_value < r_value and p_value < ag_value:
        return p, p_h, -p_value
    if r_value < p_value and r_value < ag_value:
        return r, r_h, -r_value
    return ag, ag_h, -ag_value


def shortened_step(
    x: Any,
    quadratic_form: Any,
    g_h: Any,
    p: Any,
    p_h: Any,
    lb: Any,
    ub: Any,
    theta: Any,
) -> tuple:
    """``select_step``'s first candidate alone: ``p``, or ``p`` shortened to stay inside.

    The step that keeps the subproblem's equality rows: a multiple of ``p``.

    Returns:
        ``(step, step_h, predicted_reduction)`` as ``select_step`` returns them.
    """
    x = np.asarray(x)
    p = np.array(p, copy=True)
    p_h = np.array(p_h, copy=True)
    with np.errstate(all="ignore"):
        if not in_bounds(x + p, lb, ub):
            p_stride, _ = step_size_to_bound(x, p, lb, ub)
            p *= p_stride
            p_h *= p_stride
            p *= theta
            p_h *= theta
        return p, p_h, -evaluate_quadratic(quadratic_form, g_h, p_h)


def step_back_ratio(g: Any, v: Any) -> Any:
    """``theta = max(0.995, 1 - ||g * v||_inf)``: how close to a side a step may go."""
    product = np.asarray(g) * np.asarray(v)
    largest = float(np.max(np.abs(product))) if product.size else 0.0
    return max(0.995, 1.0 - largest)


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

    v: Any
    dv: Any
    scale: Any
    addition: Any
    theta: Any


def coleman_li_scaling(x: Any, g: Any, lb: Any, ub: Any, scale: Any) -> Scaling:
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
        The [Scaling][diff_soup.inference.hierarchical._bounds.Scaling].
    """
    scale = np.asarray(scale)
    g = np.asarray(g)
    v, dv = CL_scaling_vector(x, g, lb, ub)
    theta = step_back_ratio(g, v)
    v_hat = v.copy()
    bounded = dv != 0
    v_hat[bounded] *= scale[bounded]
    with np.errstate(divide="ignore", invalid="ignore"):
        coordinates = scale / np.sqrt(v_hat)
        addition = np.where(bounded, g * dv / scale, 0.0).astype(scale.dtype)
    return Scaling(v=v, dv=dv, scale=coordinates, addition=addition, theta=theta)


def active_bounds(x: Any, bounds: Any, rtol: Any = 1e-10) -> tuple:
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
