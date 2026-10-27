"""Restarted GMRES with right preconditioning, in JAX.

[gmres_core][diff_soup.linear._krylov._gmres.gmres_core] solves ``M x = b``
for any number of independent systems side by side (a leading batch shape on
``b``), given the action ``mv`` of ``M`` and a preconditioner ``P ~ M``:

* The iteration runs on the weighted system. With ``W = diag(w)`` and
  ``D = W / sqrt(n)`` (``w = 1`` without weights), the Arnoldi process builds
  an orthonormal basis ``V`` of the Krylov space of ``D M P^-1 W^-1`` from
  ``D r0``, ``r0 = b - M x0``, and the update is ``x = x0 + P^-1 W^-1 V y``.
  The basis lives in the weighted space, so the weighting is undone before
  the preconditioner applies: with an exact preconditioner (``P = M``) the
  iterated operator is a multiple of the identity and one iteration solves
  the system whatever the weights. The least-squares residual that the Givens
  rotations update at every step is exactly ``||D (b - M x)||_2``, the
  weighted root-mean-square norm of the true residual
  (``sqrt(mean_i (w_i r_i)^2)``), so the stopping test needs no extra operator
  application.
* Classical Gram-Schmidt with one reorthogonalisation: two passes, each one
  fused inner product with the whole basis and one update.
* A static restart length ``tol.restart`` (the basis is ``restart + 1``
  vectors per system), cycles restarted from the true residual while any
  system is unconverged and its iteration count is below ``tol.max_iter``.
  Within a cycle the loop stops as soon as every system is done, so no
  preconditioner is applied for finished systems after that.
* Every system has its own done flag: a finished system keeps its state, and
  every division is guarded so the arithmetic on finished systems stays
  finite.

Stop and status per system (a [SolveStatus][diff_soup.linear.SolveStatus]):
``SUCCESS`` when ``||D (b - M x)||_2 <= atol + rtol * ||D b||_2`` holds for the
true residual recomputed at the end of a cycle; ``MAX_ITERATIONS`` at the
iteration limit; ``BREAKDOWN`` when the Arnoldi process cannot extend the
basis (``h_{j+1,j}`` zero to working precision) without convergence;
``STAGNATION`` after two consecutive cycles that each reduce the residual by
less than 10 %; ``NONFINITE`` for a non-finite right-hand side or operator
action; and, for a preconditioner application that fails, the status the
preconditioner reports when it is ``SINGULAR``, ``NONFINITE`` or ``EMPTY``
(its factor failed or holds nothing), else ``PRECONDITIONER_FAILURE`` (any
other status, or a non-finite result with a ``SUCCESS`` status). A system
whose status is one of the failures of the preconditioner or ``NONFINITE`` has
a ``nan`` solution, as a failed exact solve does; one that stopped short
(``MAX_ITERATIONS``, ``STAGNATION``, ``BREAKDOWN``) returns its last iterate.

Counts per system: ``iterations`` is the number of Arnoldi steps taken (the
preconditioner is applied once per iteration); ``applications`` counts the
applications of ``M``: one per iteration, one per cycle for the true residual,
and one for the initial residual when an initial guess is given.

Systems are independent: a block of right-hand sides is solved column by
column side by side, each column with its own Krylov space (block Krylov
methods, which share one space among the columns, are not implemented).
[gmres][diff_soup.linear.gmres] is the public entry for a
[LinearOperator][diff_soup.linear.LinearOperator].
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from diff_soup.linear._api import _bind_solve, _with_order
from diff_soup.linear._factor import (
    Factor,
    FactorContractError,
    SolveReport,
    SolveStatus,
    Tolerance,
    check_tolerance,
)
from diff_soup.linear._operator import LinearOperator

__all__ = ["gmres", "gmres_core"]

_SUCCESS = int(SolveStatus.SUCCESS)
#: Statuses of a failed preconditioner application that a solve reports as they are.
_PASSED_THROUGH = (
    int(SolveStatus.SINGULAR),
    int(SolveStatus.NONFINITE),
    int(SolveStatus.EMPTY),
)
#: Statuses whose solution is ``nan``.
_NAN_STATUSES = (
    int(SolveStatus.SINGULAR),
    int(SolveStatus.NONFINITE),
    int(SolveStatus.EMPTY),
    int(SolveStatus.PRECONDITIONER_FAILURE),
)
#: A cycle that keeps more than this fraction of the residual counts towards stagnation.
_STAGNATION_RATIO = 0.9
#: Consecutive stagnating cycles that stop a system.
_STAGNATION_CYCLES = 2


def _safe_divide(numerator: Array, denominator: Array) -> Array:
    """``numerator / denominator`` where the denominator is nonzero, else zero."""
    nonzero = denominator != 0
    return jnp.where(nonzero, numerator / jnp.where(nonzero, denominator, 1), 0)


def _isin(status: Array, codes: tuple) -> Array:
    """Whether every entry of ``status`` is one of ``codes``."""
    found = jnp.zeros(status.shape, jnp.bool_)
    for code in codes:
        found = found | (status == code)
    return found


def gmres_core(
    mv: Callable[[Array], Array],
    b: ArrayLike,
    *,
    precondition: Optional[Callable[[Array], Any]],
    tol: Tolerance,
    weights: Optional[ArrayLike] = None,
    x0: Optional[ArrayLike] = None,
    order: Any = None,
) -> tuple[Array, Array, Array, Array, Array, Any]:
    """Restarted, right-preconditioned GMRES on independent systems side by side.

    See the module docstring for the method, the stopping rule, the statuses
    and the counts.

    Args:
        mv: ``x -> M x`` on arrays of ``b``'s shape (the last axis is the
            unknowns, the leading axes index independent systems).
        b: Right-hand sides ``(*batch, n)``.
        precondition: ``v -> P^-1 v`` on arrays of ``b``'s shape, returning
            the result or ``(result, status)`` with an ``int32`` status that
            broadcasts against ``b``'s batch shape; ``None`` for no
            preconditioner. With ``order`` given it is called as
            ``precondition(v, order)`` and returns ``(result, status,
            order)``: the order word of a preconditioner that solves with a
            factor, carried from application to application through the
            iteration.
        tol: The [Tolerance][diff_soup.linear.Tolerance].
        weights: The error weights, real, of ``b``'s shape or ``(n,)``;
            ``None`` for ones.
        x0: The initial guess of ``b``'s shape; ``None`` for zero.
        order: ``None``, or the order word the first preconditioner
            application takes.

    Returns:
        ``(x, status, iterations, applications, residual, order)``: the
        solution in ``b``'s shape and dtype, then per system the ``int32``
        status, iteration and application counts and the achieved weighted
        root-mean-square residual (``nan`` for a ``nan`` solution), and the
        order word of the last preconditioner application (``order``
        unchanged when none ran, ``None`` without ``order``).

    Raises:
        FactorContractError: For an invalid tolerance or a right-hand side
            that is not an array of at least one dimension.
    """
    tol = check_tolerance(tol)
    if tol is None:
        raise FactorContractError(
            "gmres needs a stopping rule: pass tol=diff_soup.linear.Tolerance(rtol, atol, "
            "max_iter, restart)."
        )
    b = jnp.asarray(b)
    if b.ndim < 1:
        raise FactorContractError(
            f"gmres needs a right-hand side with the unknowns on its last axis; got shape "
            f"{tuple(b.shape)}."
        )
    shape, n = tuple(b.shape[:-1]), int(b.shape[-1])
    dtype = b.dtype
    real = jnp.zeros((), dtype).real.dtype
    m, max_iter = int(tol.restart), int(tol.max_iter)
    eps = jnp.asarray(jnp.finfo(real).eps, real)
    root_n = 1.0 / math.sqrt(max(n, 1))
    if weights is None:
        scale = jnp.full(b.shape, root_n, real)
        weight = None
    else:
        weight = jnp.broadcast_to(jnp.asarray(weights).astype(real), b.shape)
        scale = weight * root_n

    def norm(v: Array) -> Array:
        """``||D v||_2`` per system."""
        return jnp.sqrt(jnp.sum(jnp.abs(scale * v) ** 2, axis=-1))

    def apply_operator(v: Array) -> Array:
        """``M v`` in the dtype of ``b``."""
        return jnp.asarray(mv(v)).astype(dtype)

    def apply_preconditioner(v: Array, order: Any) -> tuple[Array, Array, Any]:
        """``(P^-1 v, status, order)`` with the status broadcast to the batch shape."""
        if precondition is None:
            return v, jnp.zeros(shape, jnp.int32), order
        if order is None:
            out = precondition(v)
        else:
            *out, order = precondition(v, order)
            out = tuple(out)
        if isinstance(out, tuple):
            z, status = out
        else:
            z, status = out, jnp.zeros(shape, jnp.int32)
        status = jnp.asarray(status, jnp.int32)
        status = jnp.reshape(status, status.shape + (1,) * (len(shape) - status.ndim))
        return jnp.asarray(z).astype(dtype), jnp.broadcast_to(status, shape), order

    threshold = tol.atol + tol.rtol * norm(b)
    zero_i = jnp.zeros(shape, jnp.int32)
    if x0 is None:
        x = jnp.zeros_like(b)
        r = b
        applications = zero_i
    else:
        x = jnp.broadcast_to(jnp.asarray(x0).astype(dtype), b.shape)
        r = b - apply_operator(x)
        applications = zero_i + 1
    rho = norm(r)
    nonfinite = ~jnp.isfinite(rho) | ~jnp.all(jnp.isfinite(b), axis=-1)
    converged = (rho <= threshold) & ~nonfinite
    status = jnp.where(nonfinite, int(SolveStatus.NONFINITE), _SUCCESS)
    if max_iter == 0:
        status = jnp.where(converged | nonfinite, status, int(SolveStatus.MAX_ITERATIONS))
        done = jnp.ones(shape, jnp.bool_)
    else:
        done = converged | nonfinite
    status = status.astype(jnp.int32)

    def cycle(carry: tuple) -> tuple:
        """One restart cycle on every system that is not done."""
        x, r, rho, done, status, iterations, applications, stagnant, count, order = carry
        start = ~done
        v0 = _safe_divide(scale * r, rho[..., None].astype(dtype))
        basis = jnp.zeros((m + 1,) + b.shape, dtype).at[0].set(jnp.where(start[..., None], v0, 0))
        directions = jnp.zeros((m,) + b.shape, dtype)
        upper = jnp.zeros((m, m) + shape, dtype)
        cosines = jnp.ones((m,) + shape, real)
        sines = jnp.zeros((m,) + shape, dtype)
        g = jnp.zeros((m + 1,) + shape, dtype).at[0].set(rho.astype(dtype))
        inner0 = (
            jnp.asarray(0, jnp.int32),
            basis,
            directions,
            upper,
            cosines,
            sines,
            g,
            start,
            zero_i,
            iterations,
            applications,
            zero_i,
            jnp.zeros(shape, jnp.bool_),
            jnp.zeros(shape, jnp.bool_),
            order,
        )

        def more(state: tuple) -> Array:
            """Whether a step remains in this cycle for some system."""
            return (state[0] < m) & jnp.any(state[7])

        def step(state: tuple) -> tuple:
            """One Arnoldi step with its Givens rotation, on the active systems."""
            (
                j,
                basis,
                directions,
                upper,
                cosines,
                sines,
                g,
                active,
                steps,
                iterations,
                applications,
                failure,
                nonfinite,
                broken,
                order,
            ) = state
            v = jnp.where(active[..., None], basis[j], 0)
            if weight is not None:
                # The basis vector is a weighted-space vector; the preconditioner acts on
                # unweighted vectors, so W^-1 is applied first (a zero weight gives zero).
                v = _safe_divide(v, weight.astype(dtype))
            z, pstatus, order = apply_preconditioner(v, order)
            failed = active & ((pstatus != _SUCCESS) | ~jnp.all(jnp.isfinite(z), axis=-1))
            code = jnp.where(
                _isin(pstatus, _PASSED_THROUGH), pstatus, int(SolveStatus.PRECONDITIONER_FAILURE)
            )
            usable = active & ~failed
            z = jnp.where(usable[..., None], z, 0)
            w = scale * apply_operator(z)
            bad = usable & ~jnp.all(jnp.isfinite(w), axis=-1)
            ok = usable & ~bad
            w = jnp.where(ok[..., None], w, 0)
            size = jnp.sqrt(jnp.sum(jnp.abs(w) ** 2, axis=-1))
            first = jnp.sum(jnp.conj(basis) * w, axis=-1)
            w = w - jnp.sum(first[..., None] * basis, axis=0)
            second = jnp.sum(jnp.conj(basis) * w, axis=-1)
            w = w - jnp.sum(second[..., None] * basis, axis=0)
            beta = jnp.sqrt(jnp.sum(jnp.abs(w) ** 2, axis=-1))
            column = (first + second).at[j + 1].set(beta.astype(dtype))

            def rotate(i: Array, col: Array) -> Array:
                c, s = cosines[i].astype(dtype), sines[i]
                top, below = col[i], col[i + 1]
                col = col.at[i].set(c * top + s * below)
                return col.at[i + 1].set(-jnp.conj(s) * top + c * below)

            column = jax.lax.fori_loop(0, j, rotate, column)
            a = column[j]
            magnitude = jnp.abs(a)
            radius = jnp.sqrt(magnitude**2 + beta**2)
            c = jnp.where(radius > 0, _safe_divide(magnitude, radius), 1.0).astype(real)
            phase = jnp.where(magnitude > 0, _safe_divide(a, magnitude.astype(dtype)), 1)
            s = phase * _safe_divide(beta, radius).astype(dtype)
            column = column.at[j].set(phase * radius.astype(dtype)).at[j + 1].set(0)
            g_new = g.at[j].set(c.astype(dtype) * g[j]).at[j + 1].set(-jnp.conj(s) * g[j])
            estimate = jnp.abs(g_new[j + 1])
            # A column that rotates to zero (the operator maps the new direction into the
            # basis it already spans) makes the least-squares problem singular: it does not
            # enter the update, and the system stops with a breakdown.
            degenerate = ok & (radius <= eps * size)
            take = ok & ~degenerate

            basis = basis.at[j + 1].set(
                jnp.where(take[..., None], _safe_divide(w, beta[..., None].astype(dtype)), 0)
            )
            directions = directions.at[j].set(jnp.where(take[..., None], z, directions[j]))
            upper = upper.at[:, j].set(jnp.where(take, column[:m], upper[:, j]))
            cosines = cosines.at[j].set(jnp.where(take, c, cosines[j]))
            sines = sines.at[j].set(jnp.where(take, s, sines[j]))
            g = jnp.where(take, g_new, g)
            steps = steps + take.astype(jnp.int32)
            iterations = iterations + active.astype(jnp.int32)
            applications = applications + usable.astype(jnp.int32)
            reached = take & (estimate <= threshold)
            breaks = ok & ~reached & ((beta <= eps * size) | degenerate)
            stop = failed | bad | reached | breaks | (iterations >= max_iter)
            failure = jnp.where(failed, code, failure).astype(jnp.int32)
            return (
                j + 1,
                basis,
                directions,
                upper,
                cosines,
                sines,
                g,
                active & ~stop,
                steps,
                iterations,
                applications,
                failure,
                nonfinite | bad,
                broken | breaks,
                order,
            )

        (
            _,
            _,
            directions,
            upper,
            _,
            _,
            g,
            _,
            steps,
            iterations,
            applications,
            failure,
            bad,
            broken,
            order,
        ) = jax.lax.while_loop(more, step, inner0)

        # The least-squares update from the steps each system took.
        triangle = jnp.moveaxis(upper, (0, 1), (-2, -1))
        rhs = jnp.moveaxis(g[:m], 0, -1)
        used = jnp.arange(m) < steps[..., None]
        square = used[..., :, None] & used[..., None, :]
        eye = jnp.eye(m, dtype=dtype)
        triangle = jnp.where(square, triangle, eye)
        diagonal = jnp.diagonal(triangle, axis1=-2, axis2=-1)
        triangle = triangle + jnp.where(diagonal == 0, 1, 0)[..., None] * eye
        rhs = jnp.where(used, rhs, 0)
        y = jax.scipy.linalg.solve_triangular(triangle, rhs[..., None], lower=False)[..., 0]
        y = jnp.where(jnp.isfinite(y), y, 0)
        x_new = x + jnp.einsum("k...n,...k->...n", directions, y)
        r_new = b - apply_operator(x_new)
        rho_new = norm(r_new)
        applications = applications + start.astype(jnp.int32)

        reached = rho_new <= threshold
        bad = bad | ~jnp.isfinite(rho_new)
        stagnant = jnp.where(rho_new > _STAGNATION_RATIO * rho, stagnant + 1, 0).astype(jnp.int32)
        outcome = jnp.select(
            [
                reached & (failure == 0),
                failure != 0,
                bad,
                broken,
                iterations >= max_iter,
                stagnant >= _STAGNATION_CYCLES,
            ],
            [
                jnp.full(shape, _SUCCESS, jnp.int32),
                failure,
                jnp.full(shape, int(SolveStatus.NONFINITE), jnp.int32),
                jnp.full(shape, int(SolveStatus.BREAKDOWN), jnp.int32),
                jnp.full(shape, int(SolveStatus.MAX_ITERATIONS), jnp.int32),
                jnp.full(shape, int(SolveStatus.STAGNATION), jnp.int32),
            ],
            default=jnp.full(shape, -1, jnp.int32),
        )
        finished = start & (outcome >= 0)
        keep = ~start
        return (
            jnp.where(keep[..., None], x, x_new),
            jnp.where(keep[..., None], r, r_new),
            jnp.where(keep, rho, rho_new),
            done | finished,
            jnp.where(finished, outcome, status).astype(jnp.int32),
            iterations,
            applications,
            jnp.where(keep, 0, stagnant).astype(jnp.int32),
            count + 1,
            order,
        )

    def unfinished(carry: tuple) -> Array:
        """Whether a cycle remains: some system is not done (cycles are capped as a guard)."""
        return jnp.any(~carry[3]) & (carry[8] <= max_iter)

    carry0 = (
        x,
        r,
        rho,
        done,
        status,
        zero_i,
        applications,
        zero_i,
        jnp.asarray(0, jnp.int32),
        order,
    )
    x, _, rho, done, status, iterations, applications, _, _, order = jax.lax.while_loop(
        unfinished, cycle, carry0
    )
    status = jnp.where(done, status, int(SolveStatus.MAX_ITERATIONS)).astype(jnp.int32)
    failed = _isin(status, _NAN_STATUSES)
    x = jnp.where(failed[..., None], jnp.nan, x).astype(dtype)
    rho = jnp.where(failed, jnp.nan, rho)
    return x, status, iterations, applications, rho, order


def gmres(
    operator: LinearOperator,
    b: ArrayLike,
    *,
    precondition: Any = None,
    tol: Tolerance,
    weights: Optional[ArrayLike] = None,
    x0: Optional[ArrayLike] = None,
    transpose: bool = False,
) -> tuple[Array, SolveReport]:
    """Solve `M x = b` with restarted GMRES on a matrix-free operator.

    Args:
        operator: The `LinearOperator` `M`.
        b: The right-hand side `(n,)`, or `(nrhs, n)` for independent systems.
        precondition: A `Factor`, a function `v -> P^-1 v`, or `None` for no preconditioner.
        tol: The `Tolerance` that stops the iteration.
        weights: The error weights of the stopping rule's root-mean-square norm.
        x0: The initial guess, zeros by default.
        transpose: Solve `M^T x = b`, which needs the operator's `rmv`.

    Returns:
        The solution and a `SolveReport`, the worst over the right-hand sides of a block.
    """
    if not isinstance(operator, LinearOperator):
        raise FactorContractError(
            "gmres needs a diff_soup.linear.LinearOperator (build one with operator_from_values "
            f"or operator_from_jvp); got {type(operator).__name__}."
        )
    b = jnp.asarray(b)
    b = b.astype(jnp.result_type(b.dtype, operator.dtype, jnp.float32))
    n = int(operator.shape[1 if not transpose else 0])
    if b.ndim not in (1, 2) or b.shape[-1] != n or operator.shape[0] != operator.shape[1]:
        raise FactorContractError(
            f"gmres needs a square operator and a right-hand side of shape ({n},) or (nrhs, {n}); "
            f"got operator shape {operator.shape} and right-hand side shape {tuple(b.shape)}."
        )
    action = operator.rmv if transpose else operator.mv
    if action is None:
        operator.apply(b, transpose=True)  # raises the operator's own error
    block = b.ndim == 2
    mv = jax.vmap(action) if block else action
    order = None
    if precondition is None:
        apply = None
    elif isinstance(precondition, Factor):
        factor = precondition
        order = factor.order

        def apply(v: Array, word: Array) -> Any:
            x, status, *_, solved = _bind_solve(
                _with_order(factor, word),
                v,
                transpose=transpose,
                tol=None,
                weights=None,
                x0=None,
                caller="gmres",
            )
            return x, status, solved.order

    elif callable(precondition):
        apply = jax.vmap(precondition) if block else precondition
    else:
        raise FactorContractError(
            "gmres takes precondition=None, a diff_soup.linear.Factor or a callable v -> P^-1 v; "
            f"got {type(precondition).__name__}."
        )
    weights = None if weights is None else jnp.broadcast_to(jnp.asarray(weights), b.shape)
    x0 = None if x0 is None else jnp.broadcast_to(jnp.asarray(x0), b.shape)
    x, status, iterations, applications, residual, _ = gmres_core(
        mv, b, precondition=apply, tol=tol, weights=weights, x0=x0, order=order
    )
    if block:
        status, iterations = jnp.max(status), jnp.max(iterations)
        applications, residual = jnp.max(applications), jnp.max(residual)
    report = SolveReport(
        status=status, residual=residual, iterations=iterations, applications=applications
    )
    return x, report
