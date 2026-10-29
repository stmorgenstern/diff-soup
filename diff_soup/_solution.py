"""The terminal solver status ``RESULT`` and its two helpers.

``RESULT`` is the single terminal-status enumeration every integrator and
nonlinear solve in the package reports. [is_successful][diff_soup._solution.is_successful]
is its success predicate and [result_from_code][diff_soup._solution.result_from_code]
maps the engines' integer termination code to a member.

**The loud-status contract.** Step-size collapse and partial results are
first-class failure statuses: ``t_reached < t_end`` can never look like
success. Only ``SUCCESS`` and ``EVENT_STOP`` are successful.
"""

from __future__ import annotations

import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


class RESULT(eqxi.Enumeration):  # pyright: ignore[reportGeneralTypeIssues]
    """The outcome of a solve, read from `Solution.result`.

    `SUCCESS` and `EVENT_STOP` are the successful outcomes, and `Solution.ok` tests for either. Compare members with `==`, as in `sol.result == ds.RESULT.MAX_STEPS`.

    Attributes:
        SUCCESS: The solve reached the end of the interval, or a nonlinear solve converged.
        EVENT_STOP: A terminating event stopped the solve.
        MAX_STEPS: The step budget ran out first.
        STEP_COLLAPSE: The step size fell below its lower bound.
        NONLINEAR_DIVERGED: The nonlinear corrector diverged.
        LINEAR_FAIL: A linear solve failed.
        SINGULAR: The operator or its factor was detected singular.
        INCONSISTENT_IC: The initial state violates the algebraic constraints.
        ACCURACY_LIMIT: The tolerances are tighter than double precision resolves at the solution's magnitude.
        UNKNOWN: The solve did not reach a terminal status.
    """

    SUCCESS = "Solve reached t_end / converged."
    EVENT_STOP = "Stopped by a terminating event."
    MAX_STEPS = "Maximum step count reached before t_end."
    STEP_COLLAPSE = "Step size collapsed below h_min before t_end."
    NONLINEAR_DIVERGED = "Nonlinear corrector diverged."
    LINEAR_FAIL = "Linear solve failed."
    SINGULAR = "Operator / factor detected singular."
    INCONSISTENT_IC = (
        "Initial state violates algebraic/Schur constraints. Under float32 compute, the "
        "residual's own rounding can exceed the initialization tolerance; "
        "Precision.fp32_ir() or Precision.fp64() initializes in float64."
    )
    ACCURACY_LIMIT = (
        "Requested tolerances are tighter than the working precision can resolve at "
        "the current solution's magnitude; loosen atol/rtol."
    )
    UNKNOWN = "Incomplete / unknown."


# equinox prints an item as ``<enumeration.__module__>.RESULT<...>``; show the public path.
RESULT.__module__ = "diff_soup"


#: ``RESULT`` member of each integer termination code, indexed by the code.
_BY_CODE = (
    RESULT.UNKNOWN,
    RESULT.SUCCESS,
    RESULT.STEP_COLLAPSE,
    RESULT.NONLINEAR_DIVERGED,
    RESULT.MAX_STEPS,
    RESULT.EVENT_STOP,
    RESULT.INCONSISTENT_IC,
    RESULT.ACCURACY_LIMIT,
)


def is_successful(result: RESULT) -> Array:
    """Whether ``result`` is a successful outcome: ``SUCCESS`` or ``EVENT_STOP``.

    Every partial or collapsed terminal status reads as failure. Traced when
    ``result`` is traced; batched results give a batched predicate.

    Args:
        result: A [RESULT][diff_soup._solution.RESULT] member (possibly traced).

    Returns:
        A boolean array of the result's shape.

    Raises:
        ValueError: If ``result`` is a member of another enumeration.
    """
    return jnp.logical_or(result == RESULT.SUCCESS, result == RESULT.EVENT_STOP)


def result_from_code(code: ArrayLike) -> RESULT:
    """Map an integer termination code to its ``RESULT`` member.

    The mapping::

        0 -> UNKNOWN              4 -> MAX_STEPS
        1 -> SUCCESS              5 -> EVENT_STOP
        2 -> STEP_COLLAPSE        6 -> INCONSISTENT_IC
        3 -> NONLINEAR_DIVERGED   7 -> ACCURACY_LIMIT

    Codes outside ``0-7`` map to ``UNKNOWN``. Code ``2`` (step-size collapse
    before ``t_end``) is a failure, so a partial solve never reads as
    success. ``code`` may be traced or batched; the member has its shape.

    Args:
        code: Integer termination code (Python int or integer array).

    Returns:
        The corresponding ``RESULT`` member.
    """
    code = jnp.asarray(code)
    table = jnp.stack([jax.tree_util.tree_leaves(member)[0] for member in _BY_CODE])
    known = (code >= 0) & (code < len(_BY_CODE))
    value = jnp.where(known, table[jnp.clip(code, 0, len(_BY_CODE) - 1)], table[0])
    return jax.tree_util.tree_map(lambda _: value, RESULT.UNKNOWN)


__all__ = ["RESULT", "is_successful", "result_from_code"]
