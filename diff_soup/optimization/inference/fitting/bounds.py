"""bounds.py — Named box bounds for optimizer decision variables.

[Bounds][diff_soup.inference.fitting.bounds.Bounds] lets a caller constrain a
[InferenceProblem][diff_soup.inference.InferenceProblem] by parameter-group name rather than
by raw flat-vector column, and
[resolve_decision_bounds][diff_soup.inference.fitting.bounds.resolve_decision_bounds]
turns that named spec into the flat ``(lb, ub)`` arrays a solver actually needs.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

_UNBOUNDED = (None, None)


@dataclasses.dataclass(frozen=True, init=False)
class Bounds:
    """Box bounds on the fitted parameters, one `(lower, upper)` pair per parameter group.

    A study builds one from the bounds declared with `spec.param(..., bounds=...)`. A `None`
    side is unbounded, and multiple-shooting window starts are never bounded.

    Args:
        spec: A mapping from group name, the handle's name or `name@member` for a per-cluster parameter, to its `(lower, upper)` pair.
        default: The `(lower, upper)` pair of every group `spec` does not name, `(None, None)` by default.
    """

    spec: tuple
    default: tuple

    def __init__(
        self,
        spec: Optional[dict[str, tuple[Optional[float], Optional[float]]]] = None,
        default: tuple[Optional[float], Optional[float]] = _UNBOUNDED,
    ):
        """Store ``spec`` as a sorted tuple of pairs so the instance stays hashable."""
        _set = object.__setattr__
        _set(self, "spec", tuple(sorted((spec or {}).items())))
        lo, hi = default
        _set(self, "default", (lo, hi))

    def items(self):
        """Return the ``(group_name, (lo, hi))`` pairs, sorted by group name."""
        return self.spec


def resolve_decision_bounds(problem: Any) -> Optional[tuple[jnp.ndarray, jnp.ndarray]]:
    """Resolve ``problem.bounds`` to flat ``(lb, ub)`` arrays aligned with ``x0``.

    Each named group in ``problem.param_group_names`` is bounded by its entry
    in ``problem.bounds`` if named, else by the ``Bounds`` default; the columns
    it owns come from the parallel ``problem.param_group_cols``. Columns not
    owned by any group stay unbounded. The decision-variable window-state tail —
    the columns past the theta block, i.e. past ``len(problem.theta_names)`` —
    is always unbounded, regardless of any group or default bound.

    When the problem carries no explicit ``bounds`` but its problem declared
    parameter bounds on the spec (``spec.param(..., bounds=(lo, hi))``), the
    problem's pre-resolved declared bounds are returned instead — an explicit
    ``Bounds`` always overrides the declarations wholesale.

    Args:
        problem: An object exposing ``bounds``, ``x0``, ``theta_names``,
            ``param_group_names``, and ``param_group_cols``.

    Returns:
        ``None`` if the problem has neither explicit nor spec-declared
        bounds; otherwise a ``(lb, ub)`` pair of ``jax`` arrays of length
        ``problem.x0.size``, with ``-inf``/``inf`` on unbounded sides.

    Raises:
        ValueError: If ``bounds`` names a parameter group the problem does not
            have, or if a resolved lower bound is not strictly below its upper bound.
    """
    bounds = getattr(problem, "bounds", None)
    if bounds is None:
        declared = getattr(problem, "_declared_bounds", None)
        if declared is None:
            return None
        declared_lb, declared_ub = declared
        if bool(jnp.all(jnp.isneginf(declared_lb))) and bool(jnp.all(jnp.isposinf(declared_ub))):
            return None
        return declared

    n = int(np.asarray(problem.x0).size)
    n_theta = len(problem.theta_names)
    lb = np.full(n, -np.inf)
    ub = np.full(n, np.inf)

    group_names = problem.param_group_names
    group_cols = problem.param_group_cols
    spec = dict(bounds.items())

    unknown = sorted(set(spec) - set(group_names))
    if unknown:
        raise ValueError(
            f"Bounds names parameter group(s) {unknown} that this problem does "
            f"not have; valid parameter group names are {sorted(group_names)}."
        )

    for gname, cols in zip(group_names, group_cols):
        lo, hi = spec.get(gname, bounds.default)
        idx = np.asarray(cols, dtype=int)
        if idx.size == 0:
            continue
        lb[idx] = -np.inf if lo is None else float(lo)
        ub[idx] = np.inf if hi is None else float(hi)

    # The window-state tail is never part of any parameter group's columns, but
    # force it unbounded explicitly so this invariant does not depend on that.
    if n > n_theta:
        lb[n_theta:] = -np.inf
        ub[n_theta:] = np.inf

    if np.any(lb >= ub):
        raise ValueError(
            "Bounds resolved a lower bound that is not strictly below its upper "
            "bound; every bounded decision variable needs lb < ub."
        )
    if np.all(np.isneginf(lb)) and np.all(np.isposinf(ub)):
        return None
    return jnp.asarray(lb), jnp.asarray(ub)


def projected_gradient_norm(x, grad, lb, ub):
    """Norm of the projected gradient ``||x - clip(x - grad, lb, ub)||``.

    This is the box-constrained stationarity measure: it equals the ordinary
    gradient norm on the interior and vanishes exactly at a constrained optimum
    on an active face, where the ordinary gradient norm does not. Solvers use it
    in place of the gradient-norm stopping test whenever bounds are active.

    Args:
        x: Current iterate.
        grad: Gradient of the objective at ``x``.
        lb: Lower bounds (``-inf`` where unbounded).
        ub: Upper bounds (``inf`` where unbounded).

    Returns:
        A scalar ``jax`` array with the projected-gradient norm.
    """
    x = jnp.asarray(x)
    grad = jnp.asarray(grad)
    return jnp.linalg.norm(x - jnp.clip(x - grad, lb, ub))
