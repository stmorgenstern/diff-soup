"""Iterative refinement of a factor's solve against full-precision operator values.

A factor built at the ``"fp32"`` precision rung factors and solves at roughly
twice the speed and half the memory of an ``"fp64"`` factor, but its solution
is accurate only to single precision. [refine][diff_soup.linear.refine]
recovers a double-precision solution from it with the classical
mixed-precision sweep

    x_0     = f^{-1} b
    r_k     = b - A x_k                 (in the dtype of ``values``)
    x_{k+1} = x_k + f^{-1} r_k

where ``A`` is the operator that ``values`` describe on the factor's pattern,
with the topology's Dirichlet rows applied exactly as the factor applies them.
The residual and the accumulation run in the dtype of ``values`` (``float64``
for full-precision values); only the corrections come from the factor.

The sweep is adaptive (a ``lax.while_loop``). It stops

* when the relative residual ``||b - A x|| / ||b||`` reaches
  ``residual_target`` (a right-hand-side block stops when its worst row
  does);
* when a step fails to halve the residual: each step contracts the residual
  by about ``cond(A) * eps32``, so a weaker contraction (or growth) means the
  remaining steps cannot reach the target, and the sweep stops at once;
* after ``max_steps`` corrections.

A solution whose final residual is above ``residual_target``, or not finite,
is reported as stalled in the [RefineReport][diff_soup.linear.RefineReport].
``refine`` never factors anything itself: a caller that needs a solution of
full accuracy whatever the conditioning reacts to ``stalled``, for example by
refactoring at the ``"fp64"`` rung and solving there.

``max_steps=0`` measures the residual of the factor's own solution and
applies no correction; with a loose ``residual_target`` (for example
[DIRECT_RESIDUAL_THRESHOLD][diff_soup.linear._refine.DIRECT_RESIDUAL_THRESHOLD])
it is the acceptance test of a direct reduced-precision solve.

The derivative of ``refine`` is refined too. The solution ``x`` solves
``A x = b`` for the operator of ``values``, so its tangent is
``A^-1 (db - dA x)``; ``refine`` computes it as

    dx, _ = refine_fixed(f, db - dA x, values=values, steps=max_steps)

that is, the factor's solve of the tangent's right-hand side followed by
``max_steps`` fixed corrections against ``values``. The rule is linear in
``db`` and ``dA``, so reverse mode transposes it (transposed solves with the
same factor and transposed corrections) and higher derivatives compose; no
derivative ever factors and the adaptive sweep itself is never
differentiated. First derivatives flow through ``b`` and ``values``; the
factor ``f`` acts as a fixed approximate inverse and its own values carry no
first derivative. On a single-precision factor each correction contracts the
error of the tangent by about ``cond(A) * eps32``, as it contracts the error
of the solution, so the derivative reaches double precision wherever the
solution does; with ``max_steps=0`` the derivative is the factor's own solve.

The adaptive stop makes ``refine`` a nonlinear function of the right-hand
side, which JAX cannot transpose. Where the right-hand side is itself a
tangent inside a custom differentiation rule,
[refine_fixed][diff_soup.linear._refine.refine_fixed] applies a fixed number
of the same corrections and stays linear in it. Its corrections form their
residuals from the differentiated ``values``, so they refine the derivative
along with the solution.

Refinement presumes that ``f`` is an exact factorization of an operator
close to the one ``values`` describe; both functions refuse a factor of an
inexact (iterative) backend.

Both functions return the factor to use next beside the solution
(``x, f, report = refine(...)``, ``x, f = refine_fixed(...)``): ``f`` with the
order word of its last solve, as [solve][diff_soup.linear.solve] returns it.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from diff_soup.linear._api import (
    _check_factor,
    _check_transpose,
    _check_values,
    _with_order,
    solve,
)
from diff_soup.linear._factor import (
    Factor,
    FactorContractError,
    effective_transpose,
)
from diff_soup.linear._operator import apply_assembled
from diff_soup.linear._primitive import pattern_matvec

#: Default relative-residual target of a refinement sweep: a double-precision
#: quality solution. A well-conditioned system reaches 1e-14 to 1e-16 in one or
#: two steps; a system whose fp32 factor is too poor to refine (condition
#: numbers near 1e6 and above) stalls far above it and is reported.
DEFAULT_RESIDUAL_TARGET: float = 1e-8

#: Default ceiling on the refinement steps. The sweep stops early at the
#: target or on a stall, so this bounds the cost only for slow contractions.
DEFAULT_MAX_STEPS: int = 5

#: Acceptance threshold of a direct reduced-precision solve (``max_steps=0``):
#: the solution must reduce the relative residual below one tenth. A Newton
#: iteration absorbs the remainder as an inexact step.
DIRECT_RESIDUAL_THRESHOLD: float = 0.1

#: A refinement step must at least halve the residual; a weaker contraction
#: stops the sweep.
STALL_CONTRACTION: float = 0.5

#: Corrections of [refine_fixed][diff_soup.linear._refine.refine_fixed] by
#: default: two corrections take a single-precision solve of a well-conditioned
#: system to double-precision accuracy.
FIXED_REFINEMENT_STEPS: int = 2


class RefineReport(eqx.Module):
    """The outcome of one `refine` call.

    Attributes:
        steps: The number of corrections applied.
        stalled: `True` when the final residual is above the target or not finite.
        residual: The final relative residual `||b - M x|| / ||b||`, the worst over a block of
            right-hand sides.
    """

    steps: Array
    stalled: Array
    residual: Array


def _check_steps(max_steps: Any) -> int:
    """``max_steps`` as a non-negative Python int (it sizes the program)."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise FactorContractError(
            f"refine() needs max_steps as a non-negative Python int; got {max_steps!r}."
        )
    return int(max_steps)


def _check_target(residual_target: Any) -> float:
    """``residual_target`` as a finite positive Python float."""
    try:
        target = float(residual_target)
    except (TypeError, ValueError):
        target = math.nan
    if not (math.isfinite(target) and target > 0.0):
        raise FactorContractError(
            "refine() needs residual_target as a finite positive number; got "
            f"{residual_target!r}. A non-finite target would accept every residual."
        )
    return target


def _operator_apply(f: Factor, values: Array, x: Array, transpose: bool) -> Array:
    """``A x`` (or ``A^T x``) for the operator ``values`` describe, Dirichlet rows applied."""
    return apply_assembled(
        f.analysis, values, x, transpose=effective_transpose(f.orientation, transpose)
    )


def _held_constant(f: Factor) -> Factor:
    """``f`` with its floating-point leaves held constant under differentiation.

    Only floating-point leaves carry a derivative, so only they pass through
    ``jax.lax.stop_gradient``; integer leaves (the handle and the order word)
    are returned as they are. The order word must not pass through it: after
    a solve whose right-hand side is a tangent (inside a differentiation
    rule) the order word is part of the linear computation that reverse mode
    transposes, and ``stop_gradient`` has no transpose rule.
    """

    def hold(leaf: Any) -> Any:
        if jnp.issubdtype(jnp.result_type(leaf), jnp.inexact):
            return jax.lax.stop_gradient(leaf)
        return leaf

    return jax.tree_util.tree_map(hold, f)


def _check_exact(f: Factor, caller: str) -> None:
    """Refuse a factor of an inexact backend: refinement presumes an exact factorization."""
    if not f.exact:
        name = type(f.analysis.backend).__name__
        raise FactorContractError(
            f"{caller}() needs a factor that applies an exact inverse, but this factor comes "
            f"from the inexact backend {name} (is_exact=False). Refinement corrects an exact "
            "factorization of a nearby operator; factor with an exact backend (Dense, KLU, "
            "SuperLU or CuDSS), or read an inexact solve's convergence from solve_checked."
        )


def _sweep(
    target: float, max_steps: int, transpose: bool, f: Factor, b: Array, values: Array
) -> tuple[Array, RefineReport, Array]:
    """The adaptive refinement sweep: the refined solution, its report and the last order word."""
    x0, f = solve(f, b, transpose=transpose)
    dtype = jnp.result_type(values.dtype, b.dtype, x0.dtype)
    values = values.astype(dtype)
    b = b.astype(dtype)
    x_start = x0.astype(dtype)

    b_norm = jnp.linalg.norm(b, axis=-1)
    safe_b = jnp.where(b_norm > 0, b_norm, jnp.ones_like(b_norm))

    def relative(r: Array) -> Array:
        return jnp.max(jnp.linalg.norm(r, axis=-1) / safe_b)

    def residual(x: Array) -> Array:
        return b - _operator_apply(f, values, x, transpose)

    r_start = residual(x_start)
    rel_start = relative(r_start)
    steps = jnp.zeros((), jnp.int32)
    x_ref, rel, order = x_start, rel_start, f.order
    if max_steps > 0:

        def needs_step(carry):
            step, _x, _r, rel, escalated, _order = carry
            return (step < max_steps) & (rel > target) & jnp.logical_not(escalated)

        def step_once(carry):
            step, x, r, rel, _, order = carry
            correction, held = solve(_with_order(f, order), r, transpose=transpose)
            x_new = x + correction.astype(dtype)
            r_new = residual(x_new)
            rel_new = relative(r_new)
            escalated = rel_new > STALL_CONTRACTION * rel
            return step + 1, x_new, r_new, rel_new, escalated, held.order

        init = (steps, x_start, r_start, rel_start, jnp.zeros((), jnp.bool_), order)
        steps, x_ref, _r, rel, _escalated, order = jax.lax.while_loop(needs_step, step_once, init)

    # ``rel <= target`` is false for a nan residual, so a nan solution is stalled.
    stalled = jnp.logical_not(rel <= target)
    return x_ref, RefineReport(steps=steps, stalled=stalled, residual=rel), order


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _refined(
    target: float, max_steps: int, transpose: bool, f: Factor, b: Array, values: Array
) -> tuple[Array, RefineReport, Array]:
    """The refined solution, its report and the last order word, with the refined derivative rule."""
    return _sweep(target, max_steps, transpose, f, b, values)


def _zero_report_tangent(report: RefineReport) -> RefineReport:
    """Zero tangents of a report: ``float0`` for the integer and boolean fields."""

    def zero(leaf: Array) -> Any:
        if jnp.issubdtype(leaf.dtype, jnp.inexact):
            return jnp.zeros_like(leaf)
        return np.zeros(np.shape(leaf), dtype=jax.dtypes.float0)

    return jax.tree_util.tree_map(zero, report)


@_refined.defjvp
def _refined_jvp(target, max_steps, transpose, primals, tangents):
    """``dx = refine_fixed(f, db - dA x, values=values, steps=max_steps)``; the report has none.

    The primal comes from ``_refined`` itself, so a further derivative uses
    this rule again. ``dA`` is the tangent of ``values`` on the pattern
    with the Dirichlet rows dropped (they are constant); the factor's own
    tangent is not used. The order word has a ``float0`` tangent, and the
    order words of the tangent's solves are dropped.
    """
    f, b, values = primals
    _, db, dvalues = tangents
    x, report, order = _refined(target, max_steps, transpose, f, b, values)
    effective = effective_transpose(f.orientation, transpose)
    db, dvalues = (
        jnp.zeros(np.shape(t), x.dtype) if t.dtype == jax.dtypes.float0 else t.astype(x.dtype)
        for t in (jnp.asarray(db), jnp.asarray(dvalues))
    )
    rhs = db - pattern_matvec(f.analysis, dvalues, x, transpose=effective)
    dx, _ = refine_fixed(f, rhs, values=values, steps=max_steps, transpose=transpose)
    dorder = np.zeros(np.shape(order), dtype=jax.dtypes.float0)
    return (x, report, order), (dx.astype(x.dtype), _zero_report_tangent(report), dorder)


def refine(
    f: Factor,
    b: ArrayLike,
    *,
    values: ArrayLike,
    residual_target: float = DEFAULT_RESIDUAL_TARGET,
    max_steps: int = DEFAULT_MAX_STEPS,
    transpose: bool = False,
) -> tuple[Array, Factor, RefineReport]:
    """Solve with a factor, then refine the solution against the values of the matrix.

    Use it with a float32 factor of a float64 matrix, or a factor of a nearby matrix. It stops
    at `residual_target`, after `max_steps`, or when a step fails to halve the residual.

    Args:
        f: A factor from a direct backend.
        b: The right-hand side.
        values: The values of the matrix to solve, one per pattern entry.
        residual_target: The relative residual `||b - M x|| / ||b||` to reach, default `1e-8`.
        max_steps: The most correction steps, default `5`.
        transpose: Solve with the transpose instead.

    Returns:
        The solution, the factor to use next, and a `RefineReport`.

    Raises:
        FactorContractError: For a factor from an iterative backend.
    """
    f = _check_factor(f, "refine")
    _check_exact(f, "refine")
    transpose = _check_transpose(transpose)
    max_steps = _check_steps(max_steps)
    target = _check_target(residual_target)
    values = _check_values(f.analysis, values, "refine")
    x, report, order = _refined(target, max_steps, transpose, f, jnp.asarray(b), values)
    return x, _with_order(f, order), report


def refine_fixed(
    f: Factor,
    b: ArrayLike,
    *,
    values: ArrayLike,
    steps: int = FIXED_REFINEMENT_STEPS,
    transpose: bool = False,
) -> tuple[Array, Factor]:
    """Solve with ``f`` and apply exactly ``steps`` refinement corrections.

    The same corrections as [refine][diff_soup.linear.refine], without the
    adaptive stop or a report, so the result is a fixed linear function of
    ``b``. That makes it usable where the right-hand side is itself a
    tangent inside a custom differentiation rule: JAX can transpose it (its
    transpose is the same number of corrections on the transposed system)
    and differentiate it again.

    The corrections solve with ``f`` held constant but form their residuals
    ``b - A x`` from the differentiated ``values``, so the derivative is
    refined along with the solution. With ``f`` built from ``values`` (cast
    to the factor's precision rung), each correction contracts the error of
    the tangent against the exact ``A^-1 (db - dA x)`` as it contracts the
    error of the solution, the part from ``values`` one correction behind
    (its residual uses the previous solution). On a single-precision factor
    of a well-conditioned system one correction gives a derivative with
    respect to ``b`` accurate to double precision and one with respect to
    ``values`` accurate to single precision; two give both to double
    precision.

    Args:
        f: The [Factor][diff_soup.linear.Factor] whose solves supply the
            corrections.
        b: Right-hand side, ``(n,)`` or ``(nrhs, n)``.
        values: Operator values ``(nnz,)`` on ``f``'s pattern that define the
            system being solved.
        steps: Number of corrections, a non-negative Python int.
        transpose: Solve with the transposed operator; a Python bool.

    Returns:
        ``(x, f)``: the solution, with the shape of ``b`` in the residual
        dtype, and the factor to use next (``f`` with the order word of its
        last solve).

    Raises:
        FactorContractError: As for ``refine``.
    """
    f = _check_factor(f, "refine_fixed")
    _check_exact(f, "refine_fixed")
    transpose = _check_transpose(transpose)
    steps = _check_steps(steps)
    values = _check_values(f.analysis, values, "refine_fixed")
    b = jnp.asarray(b)
    x, f = solve(f, b, transpose=transpose)
    dtype = jnp.result_type(values.dtype, b.dtype, x.dtype)
    x = x.astype(dtype)
    if steps == 0:
        return x, f
    # Only the correction solves hold the factor constant. The residual keeps
    # the derivative of the values, so its tangent carries ``-dA x`` and the
    # corrections drive the tangent toward ``A^-1 (db - dA x)``.
    held = _held_constant(f)
    values = values.astype(dtype)
    b = b.astype(dtype)
    for _ in range(steps):
        r = b - _operator_apply(f, values, x, transpose)
        correction, held = solve(held, r, transpose=transpose)
        x = x + correction.astype(dtype)
    return x, _with_order(f, held.order)


__all__ = [
    "DEFAULT_MAX_STEPS",
    "DEFAULT_RESIDUAL_TARGET",
    "DIRECT_RESIDUAL_THRESHOLD",
    "FIXED_REFINEMENT_STEPS",
    "RefineReport",
    "STALL_CONTRACTION",
    "refine",
    "refine_fixed",
]
