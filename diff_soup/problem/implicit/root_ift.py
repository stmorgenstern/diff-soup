"""Implicit-function derivatives of accepted implicit step roots.

An accepted implicit time step is the root ``y*`` of

    G(y) = F(t, y, cj * y + q, theta, control) = 0,

where ``q`` aggregates the integration history (the accepted slope is
``ydot* = cj * y* + q``). Derivatives never flow through the Newton iteration
that found the root: the tangent solves ``A_* dy = -dG`` with the exact
sensitivity matrix ``A_* = F_y + cj * F_ydot`` evaluated at the accepted root
with the exact ``theta``, ``control``, and ``cj``. By default ``A_*`` is
rebuilt from the problem's linearization plan; an engine may instead offer a
cached iteration matrix through a static extractor that reads the corrector's
own bookkeeping and returns an [IterationMatrixCertificate][], and those
values are consumed only when every attestation holds — correctness never
depends on a cache hit.

One entry, [implicit_step_root][], a ``custom_jvp``, carries the derivative.
Its rule is linear in the tangents, so JAX derives reverse mode by
transposing it: the cotangent solve ``A_*^T lam = ybar`` factors the
transposed matrix ``A_*^T`` directly, on the reverse sweep's own precision
policy (its route, and its ``adjoint_compute_dtype`` when one is declared).
The rule computes its primal through the entry itself, so differentiating
the rule again (``jax.hessian``, reverse over forward, forward over reverse)
differentiates the root by the same rule, never through the Newton
iteration; every composition of ``jax.jvp``, ``jax.grad``, ``jax.jit`` and
``jax.vmap`` is served by this one entry.

Differentiable inputs are ``q``, ``theta``, and ``control``. The step time and
leading coefficient come from the realized adaptive schedule and are
gradient-dead, as are initial guesses and the acceptance flag. An unaccepted
primal (``ok`` false) has no supported derivative: its tangents are NaN, and
its input tangents are zeroed first, so a reverse pass gives an attempt that
is discarded (or not accepted) exact zero input cotangents even where the
residual's partials at its failed iterate are infinite. A singular
sensitivity solve gives non-finite derivatives on every backend.

With an iterative backend ([Krylov][diff_soup.linear.Krylov]) the tangent
solves run on its exact inner backend, and the cotangent solves run GMRES on
``A_*^T`` preconditioned by a factor of a transposed matrix held across the
steps of a reverse sweep: a step factors only where the held preconditioner
no longer converges within the backend's ``lagged_max_iter`` iterations.
"""

from __future__ import annotations

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import zero_from_primal

from diff_soup.linear import (
    SparseTopology,
    Tolerance,
    empty_factor,
    refactor,
    solve_checked,
)
from diff_soup.linear._context import DEFAULT_CONTEXT
from diff_soup.linear._krylov import exact_analysis, report_outcome
from diff_soup.nonlinear._kkt import linear_route, transposable_solve
from diff_soup.utils.lanes import lane_cond

__all__ = [
    "IterationMatrixCertificate",
    "certify_exact_root",
    "implicit_step_root",
    "iteration_matrix_offering",
]


class IterationMatrixCertificate(eqx.Module):
    """Attestation that cached iteration-matrix values equal the exact ``A_*``.

    ``values`` are matrix values in the layout the linearization plan
    produces: values over the union support in sparse mode, or a dense
    ``(n, n)`` matrix. The certificate is consumed only when ``exact`` is
    declared, ``at_root`` holds, and ``cj`` equals the step's leading
    coefficient exactly; any failed attestation falls back to rebuilding the
    exact matrix at the root. Certificates carry matrix values, never opaque
    backend factors, so either solve orientation can be factorized from them.

    Attributes:
        values: Candidate matrix values (union layout or dense).
        cj: The leading coefficient the values were composed with.
        at_root: Whether the values were evaluated at this accepted root.
        exact: Static declaration that the values are exact, not approximate.
    """

    values: jax.Array
    cj: jax.Array
    at_root: jax.Array
    exact: bool = eqx.field(static=True, default=True)


def iteration_matrix_offering(plan, fy_values, fydot_values, cj, at_root):
    """Build the iteration-matrix offering from a corrector's saved partials.

    Recomposes the exact combined values ``A(cj) = F_y + cj * F_ydot`` over the
    plan's union support from the corrector's saved split partials — value
    assembly only, no residual re-evaluation — and wraps them in an
    [IterationMatrixCertificate][] at the step's leading coefficient. The
    ``at_root`` predicate is the engine's strict root-success decision: the
    offering is consumed only when it holds and the coefficients match, so a
    recomposed or window-reused setup (which fails the predicate) can never
    have its stale values consumed. A failed attestation always falls back to
    rebuilding the exact matrix at the root.

    Args:
        plan: The problem's linearization plan.
        fy_values: The saved ``F_y`` split-partial values.
        fydot_values: The saved ``F_ydot`` split-partial values.
        cj: The step's exact leading coefficient.
        at_root: The engine's strict root-success predicate (traced boolean).

    Returns:
        The offering [IterationMatrixCertificate][].
    """
    values = plan.recompose(fy_values, fydot_values, cj)
    return IterationMatrixCertificate(
        values=values, cj=jnp.asarray(cj), at_root=jnp.asarray(at_root), exact=True
    )


def certify_exact_root(generations, root_ok):
    """Issue the exact-root certificate for a carry's held linearization.

    The certificate contract lives in
    [diff_soup.nonlinear.generations][]; this wrapper keeps issuance
    reachable next to the root-differentiation machinery. Call it exactly
    where a root solve's acceptance is decided, passing that decision as
    ``root_ok``.
    """
    from diff_soup.nonlinear.generations import certify_exact_root as _certify

    return _certify(generations, root_ok)


def _zero_cotangent(x):
    """Zero cotangent (or tangent) matching ``x``'s pytree and leaf avals."""
    return jax.tree_util.tree_map(lambda leaf: zero_from_primal(leaf, symbolic_zeros=False), x)


def _certificate_valid(certificate: IterationMatrixCertificate, cj) -> jax.Array:
    """Runtime attestation: evaluated at this root with this exact ``cj``."""
    return jnp.logical_and(
        jnp.asarray(certificate.at_root), jnp.asarray(certificate.cj) == jnp.asarray(cj)
    )


def _offering_from(extractor, aux, t, cj, theta, control, primal_args):
    """Build the iteration-matrix offering for a derivative rule, if any.

    A supplied static ``extractor`` builds the offering from the corrector
    ``aux`` the primal already returned, so sourcing the offering needs no
    second primal run; without one there is no offering and the sensitivity
    solve rebuilds the matrix. An offering is gradient-dead either way: the
    solve consumes only its first-derivative matrix value, and the matrix's own
    tangent always rebuilds at the frozen root.
    """
    if extractor is None:
        return None
    return extractor(aux, t, cj, theta, control, primal_args)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _offered_matrix(plan, dense, offering_values, valid, t, y, ydot, theta, control, cj):
    """The exact ``A_*`` primal from the offering, with the rebuild's tangent.

    When ``valid`` holds the offered values are returned unchanged (the value
    sweep is skipped for a first derivative); otherwise the matrix is rebuilt
    from the plan. The tangent, however, is ALWAYS the exact matrix's tangent
    at the frozen root, so a second derivative through a consumed offering
    still recovers the matrix's dependence on the differentiable inputs — a
    consumed offering changes only where a first-derivative matrix VALUE comes
    from, never the derivative it carries.
    """

    def rebuild():
        if dense:
            return plan.combined_dense(t, y, ydot, theta, control, cj)
        return plan.combined_values(t, y, ydot, theta, control, cj)

    return lane_cond(jnp.logical_not(valid), rebuild, lambda: offering_values, false_is_free=True)


@_offered_matrix.defjvp
def _offered_matrix_jvp(plan, dense, primals, tangents):
    """Primal from the offering (or rebuild); tangent from the exact rebuild."""
    offering_values, valid, t, y, ydot, theta, control, cj = primals
    _d_off, _d_valid, _dt, dy, dydot, dtheta, dcontrol, _dcj = tangents
    primal_out = _offered_matrix(plan, dense, *primals)

    def rebuild_fn(y_in, ydot_in, theta_in, control_in):
        if dense:
            return plan.combined_dense(t, y_in, ydot_in, theta_in, control_in, cj)
        return plan.combined_values(t, y_in, ydot_in, theta_in, control_in, cj)

    _, tangent_out = jax.jvp(rebuild_fn, (y, ydot, theta, control), (dy, dydot, dtheta, dcontrol))
    return primal_out, tangent_out


def _matrix_values(plan, certificate, t, y, ydot, theta, control, cj):
    """Exact ``A_*`` values over the plan's union support, honoring a valid certificate."""
    if certificate is None or not certificate.exact:
        return plan.combined_values(t, y, ydot, theta, control, cj)
    return _offered_matrix(
        plan,
        False,
        jnp.asarray(certificate.values),
        _certificate_valid(certificate, cj),
        t,
        y,
        ydot,
        theta,
        control,
        cj,
    )


def _matrix_dense(plan, certificate, t, y, ydot, theta, control, cj):
    """Exact dense ``A_*`` at the root, honoring a valid certificate."""
    if certificate is None or not certificate.exact:
        return plan.combined_dense(t, y, ydot, theta, control, cj)
    values = jnp.asarray(certificate.values)
    if values.ndim == 2:
        offered = values
    else:
        topology = plan.union_topology
        offered = (
            jnp.zeros((plan.n, plan.n), dtype=values.dtype)
            .at[np.asarray(topology.rows), np.asarray(topology.cols)]
            .add(values)
        )
    return _offered_matrix(
        plan,
        True,
        offered,
        _certificate_valid(certificate, cj),
        t,
        y,
        ydot,
        theta,
        control,
        cj,
    )


@functools.lru_cache(maxsize=64)
def _full_topology(n: int) -> SparseTopology:
    """The dense ``n x n`` pattern a dense sensitivity matrix is factored on."""
    return SparseTopology.full(n)


def _sensitivity_solve(
    plan,
    backend,
    analysis,
    reverse_analysis,
    t,
    y,
    ydot,
    theta,
    control,
    cj,
    rhs,
    *,
    certificate=None,
):
    """The tangent solve ``A_*^{-1} rhs`` at the root, linear in ``rhs``.

    The exact ``A_*`` is factored on the factor contract: with a backend and
    its analysis of the full-system pattern the solves run through that
    backend (a dense-mode plan hands it the dense matrix as row-major
    values); without one, the dense backend factors the dense matrix. The
    values stay differentiable, so a second derivative through this rule
    sees the dependence of ``A_*`` on the inputs. Solvability is not probed
    here: a singular factorization gives a non-finite solve on every
    backend, so an unsolvable sensitivity system is loud in the derivative
    itself.

    The solve is a [transposable_solve][diff_soup.nonlinear._kkt.transposable_solve]:
    evaluated, it factors ``A_*`` on the route of ``analysis``; transposed by
    JAX (a reverse sweep), it factors ``A_*^T`` directly on the route of
    ``reverse_analysis``, at that analysis's ``adjoint_compute_dtype`` when
    its precision declares one (the values are cast up before factoring). A
    precision declaring a reduced solve mode (``"fp32"`` or ``"fp32_ir"``)
    factors in single precision and refines against the double-precision
    values with a fixed number of corrections. On an iterative backend the
    forward solve runs on its exact inner backend and the transposed solve
    is the preconditioned GMRES of
    [_inexact_cotangent_solver][diff_soup.problem.implicit.root_ift._inexact_cotangent_solver].

    An offered ``certificate`` supplies the matrix values when its
    attestations hold; without one (the default) the matrix is rebuilt from
    the plan.
    """
    adjoint_dtype = getattr(
        getattr(reverse_analysis, "precision", None), "adjoint_compute_dtype", None
    )
    topology = None
    dense = backend is None or analysis is None or plan.mode != "sparse"
    if backend is None or analysis is None:
        backend = analysis = reverse_analysis = None
        topology = _full_topology(int(plan.n))

    def values_at_root():
        """The exact ``A_*`` values at the root, assembled where a solve needs them."""
        if dense:
            matrix = _matrix_dense(plan, certificate, t, y, ydot, theta, control, cj)
            return jnp.asarray(matrix).reshape(-1)
        return jnp.asarray(_matrix_values(plan, certificate, t, y, ydot, theta, control, cj))

    # The matrix is assembled inside the solves (see transposable_solve), so
    # a reverse sweep assembles it only for the steps whose pullback it
    # applies; its dtype is read from the trace alone.
    values_dtype = jax.eval_shape(values_at_root).dtype
    reverse_dtype = values_dtype if adjoint_dtype is None else jnp.dtype(adjoint_dtype)
    route = linear_route(backend, analysis, dtype=values_dtype, topology=topology)
    reverse = linear_route(backend, reverse_analysis, dtype=reverse_dtype, topology=topology)
    transpose_solve = None
    if not reverse.exact:
        inexact = reverse

        def transpose_solve(ct):
            """The cotangent solve on the iterative backend, at the reverse dtype."""
            solve = _inexact_cotangent_solver(inexact, values_at_root().astype(reverse_dtype))
            return solve(ct.astype(reverse_dtype))

        reverse = None
    if not route.exact:
        # The tangent solve is exact: it runs on the exact backend behind the
        # iterative one, on the same pattern.
        route = linear_route(None, exact_analysis(route.full), dtype=values_dtype)
    return transposable_solve(
        route,
        values_at_root,
        rhs,
        reverse_route=reverse,
        adjoint_dtype=adjoint_dtype,
        transpose_solve=transpose_solve,
    )


#: The shared storage name of the preconditioner the cotangent solves of an
#: iterative backend hold from step to step of a reverse sweep.
_REVERSE_STORAGE = "reverse-sweep"


def _inexact_cotangent_solver(route, values):
    """The cotangent solve ``x -> A_*^{-T} x`` on the route of an iterative backend.

    The factor is a [Krylov][diff_soup.linear.Krylov] factor of orientation
    ``"AT"`` whose operator is ``A_*^T`` of ``values`` and whose
    preconditioner is a primary factor of a transposed matrix held in the
    shared storage ``"reverse-sweep"`` of the default solve context: within
    one execution of a program, every step of a reverse sweep preconditions
    with the factor the storage holds (the one an earlier step built, for a
    nearby root), so most steps factor nothing. The solve first runs at most
    the backend's ``lagged_max_iter`` iterations against that preconditioner;
    where it does not converge (the storage holds nothing yet, or the
    preconditioner has grown too stale) the preconditioner is rebuilt from
    ``values`` (only in those lanes under ``vmap``) and the solve continues
    from its last iterate with the backend's full iteration budget. The
    stopping rule is the route's relative residual target
    (``adjoint_target``, else ``residual_target``) in the unweighted
    root-mean-square norm.

    The operations of one cotangent solve on the shared storage are passed
    one factor: the value check that makes the storage stand for this step's
    operator, the lagged solve, the rebuild that follows it where the lagged
    solve missed, and the continued solve. The steps of a reverse sweep are
    iterations of its loop, so one step's operations on the storage finish
    before the next step's begin.
    """
    analysis = route.full
    backend = analysis.backend
    target = route.adjoint_target if route.adjoint_target is not None else route.residual_target
    default = backend.default_tolerance
    lagged = Tolerance(target, 0.0, backend.lagged_max_iter, default.restart)
    fresh = Tolerance(target, 0.0, default.max_iter, default.restart)
    ctx = None if backend.factor_residency == "arrays" else DEFAULT_CONTEXT
    held = empty_factor(
        analysis, ctx=ctx, orientation="AT", dtype=values.dtype, storage=_REVERSE_STORAGE
    )
    current = refactor(held, values, jnp.asarray(False))

    def solve(rhs):
        """Cotangent solve against the held preconditioner, rebuilt where it fails."""
        x, solved, report = solve_checked(current, rhs, tol=lagged)
        stale = jnp.logical_not(report.converged)
        first = (x, report.status, report.iterations, report.residual)

        def rebuild(_):
            fresh_factor = refactor(solved, values, stale)
            start = jnp.where(jnp.isfinite(x), x, jnp.zeros_like(x))
            x_new, _, again = solve_checked(fresh_factor, rhs, tol=fresh, x0=start)
            return x_new, again.status, report.iterations + again.iterations, again.residual

        x, status, iterations, residual = lane_cond(
            stale, rebuild, lambda _: first, None, false_is_free=True
        )
        report_outcome("cotangent", "AT", False, status, iterations, residual, rebuilt=stale)
        return x

    return solve


def _poison(x, valid):
    """Replace ``x`` with NaN when the root or its sensitivity solve is invalid."""
    return jnp.where(valid, x, jnp.full_like(x, jnp.nan))


def _zero_where_not(keep, tree):
    """Replace every inexact leaf of ``tree`` with exact zeros where ``keep`` is false.

    A selection, not a product: an infinite or NaN leaf that is not kept
    becomes zero instead of NaN. Integer (``float0``) leaves pass unchanged.
    """
    keep = jnp.asarray(keep)

    def gate(leaf):
        if getattr(leaf, "dtype", None) == jax.dtypes.float0:
            return leaf
        return jnp.where(keep, leaf, jnp.zeros_like(leaf))

    return jax.tree_util.tree_map(gate, tree)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3, 4, 12))
def implicit_step_root(
    plan,
    backend,
    analysis,
    primal_fn,
    reverse_analysis,
    t,
    cj,
    q,
    theta,
    control,
    guess,
    primal_args,
    certificate_extractor=None,
):
    """Accepted implicit step root ``y*`` with its implicit-function derivative.

    ``primal_fn(t, cj, q, theta, control, guess, primal_args)`` returns
    ``(y_star, ok, aux)``: the converged root, whether the solve is valid, and
    any pytree of primal bookkeeping (counters, rates, carried matrix values).
    ``primal_args`` is a gradient-dead pytree of extra primal-solve state
    (error weights, carried matrices, solver memory) — thread traced values
    through it, never through the ``primal_fn`` closure, which must stay free
    of traced captures. The root carries the exact implicit-function
    derivative; ``ok``, ``aux``, and ``primal_args`` are gradient-dead.

    The derivative rule is linear in the tangents: forward mode solves with a
    factor of ``A_*`` on the route of ``analysis``, and reverse mode (JAX's
    transposition of the rule) solves with a factor of ``A_*^T`` built
    directly on the route of ``reverse_analysis``. The rule's primal runs
    through this entry, so a further differentiation (``jax.hessian``,
    ``jax.jvp`` of ``jax.grad``) differentiates the root by the rule again.

    The iteration-matrix offering that lets the sensitivity solve skip the
    value rebuild is supplied by ``certificate_extractor`` (static), an
    optional ``(aux, t, cj, theta, control, primal_args) ->
    IterationMatrixCertificate`` called inside the derivative rule on the
    ``aux`` the primal already returns, so no second primal run is needed to
    source the offering. The offering is gradient-dead and consumed only when
    every attestation holds; without an extractor the matrix is rebuilt from
    the plan.

    Args:
        plan: The problem's linearization plan (static).
        backend: Sparse linear backend for the sensitivity solves, or ``None``
            for a dense solve (always dense when the plan is dense-mode).
        analysis: Backend analysis of the plan's union support for the forward
            solves, whose precision is the forward sensitivity policy
            (``None`` when ``backend`` is ``None``).
        primal_fn: Converged root solve ``(t, cj, q, theta, control, guess,
            primal_args) -> (y_star, ok, aux)``; never differentiated.
        reverse_analysis: Backend analysis for the transposed solves of a
            reverse sweep, whose precision is the reverse sensitivity policy
            (its ``adjoint_compute_dtype`` sets the dtype of those solves), or
            ``None`` to use ``analysis``.
        t: Step time from the realized schedule (gradient-dead).
        cj: Leading slope coefficient from the realized schedule
            (gradient-dead).
        q: History aggregate; the accepted slope is ``cj * y* + q``.
        theta: Problem parameters.
        control: Control values the residual reads, or ``None``.
        guess: Initial iterate for the primal solve (gradient-dead).
        primal_args: Gradient-dead primal-solve state.
        certificate_extractor: Optional static offering extractor.

    Returns:
        ``(y_star, ok, aux)``. When ``ok`` is false the root's tangents are
        NaN.
    """
    del plan, backend, analysis, reverse_analysis, certificate_extractor
    return primal_fn(t, cj, q, theta, control, guess, primal_args)


@implicit_step_root.defjvp
def _implicit_step_root_jvp(
    plan,
    backend,
    analysis,
    primal_fn,
    reverse_analysis,
    certificate_extractor,
    primals,
    tangents,
):
    """Root tangent from the implicit-function rule; zero bookkeeping tangents."""
    t, cj, q, theta, control, guess, primal_args = primals
    _dt, _dcj, dq, dtheta, dcontrol, _dguess, _dargs = tangents
    # The primal through the entry itself: differentiating this rule again
    # differentiates the root by this rule, never through the Newton
    # iteration. Under a first derivative this is the one primal run.
    out = implicit_step_root(
        plan,
        backend,
        analysis,
        primal_fn,
        reverse_analysis,
        t,
        cj,
        q,
        theta,
        control,
        guess,
        primal_args,
        certificate_extractor,
    )
    y_star, ok, aux = out
    ydot_star = cj * y_star + q
    # An invalid root's tangent is poisoned below whatever reaches it, so its
    # input tangents are zeroed first: a reverse pass that transposes this
    # rule then gives a discarded attempt exact zero input cotangents, even
    # where the step residual's partials at its failed iterate are infinite
    # (zero times infinity would be NaN).
    dq, dtheta, dcontrol = _zero_where_not(ok, (dq, dtheta, dcontrol))
    offering = _offering_from(certificate_extractor, aux, t, cj, theta, control, primal_args)

    def step_residual(q_in, theta_in, control_in):
        """The step residual at the frozen root, as a function of its inputs."""
        return plan._residual(t, y_star, cj * y_star + q_in, theta_in, control_in)

    _, rhs_dot = jax.jvp(step_residual, (q, theta, control), (dq, dtheta, dcontrol))
    dy = _sensitivity_solve(
        plan,
        backend,
        analysis,
        analysis if reverse_analysis is None else reverse_analysis,
        t,
        y_star,
        ydot_star,
        theta,
        control,
        cj,
        -rhs_dot,
        certificate=offering,
    )
    return out, (
        _poison(dy, jnp.asarray(ok)),
        _zero_cotangent(ok),
        _zero_cotangent(aux),
    )
