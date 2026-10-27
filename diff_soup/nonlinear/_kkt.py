"""Linear solves of the nonlinear solvers and the implicit-function derivatives.

Every iteration matrix the library's nonlinear solvers and implicit-function
derivatives factor goes through the factor contract of
[diff_soup.linear][diff_soup.linear] (``analyze``, ``factor``, ``solve``)
by way of this module:

* [linear_route][diff_soup.nonlinear._kkt.linear_route] turns the
  ``(backend, analysis)`` pair a caller holds into a
  [LinearRoute][diff_soup.nonlinear._kkt.LinearRoute]: the contract analysis
  of the pattern at the precision rung of the matrix values, plus the
  reduced-precision solve policy the caller declared in its
  [Precision][diff_soup.Precision].
* [RouteFactor][diff_soup.nonlinear._kkt.RouteFactor] factors one matrix on a
  route, in the orientation the caller names (``"A"`` for the matrix,
  ``"AT"`` for its transpose), and serves its solves in both orientations
  from that one factorization.
* [transposable_solve][diff_soup.nonlinear._kkt.transposable_solve] is the
  solve of the implicit-function derivative rules: linear in its right-hand
  side, it factors the matrix for a forward tangent and factors the
  transposed matrix directly, in the reverse sweep's own precision policy,
  when JAX transposes it for a reverse-mode gradient.
* [kkt_solve][diff_soup.nonlinear._kkt.kkt_solve] and
  [kkt_solve_transposed][diff_soup.nonlinear._kkt.kkt_solve_transposed]
  eliminate a factored state block from a bordered (KKT) system.

Factors that a backend keeps in native storage live in the process-wide
default solve context, so a compiled program factors each matrix site in
storage it keeps across calls. A layout that spreads one factorization over
the devices of a ``"model"`` mesh axis is analysed on the contract like any
other; the cuDSS backend serves it in double precision.

On a reduced-precision route (solve mode ``"fp32"`` or ``"fp32_ir"``) the
derivative of a solve is a solve with the factor that produced the solution:
the single-precision factor where its refined or checked solution met the
route's target, followed on an ``"fp32_ir"`` route by the route's
``max_steps`` fixed refinement corrections against the double-precision
values (so the derivative reaches double precision where the solution does),
and the double-precision fallback factor where it did not, accurate to double
precision. Reverse mode through a solve of a held factor runs the transposed
solves of that factor only; the reverse sweep of an implicit-function
derivative ([transposable_solve][diff_soup.nonlinear._kkt.transposable_solve])
factors the transposed matrix directly instead. The
double-precision fallback factor is built outside any conditional and
refactored only in the lanes that fall back, so under ``vmap`` every
derivative solve receives its own lane's factor. The route's adjoint residual
target (``adjoint_ir_tol`` of the [Precision][diff_soup.Precision]) applies
only to the transposed solves of the implicit-function sensitivities'
backward pass; the nonlinear solvers' corrector solves never use it.

The consumers of this module (the implicit-function derivatives, the
consistent initialisation, the projections and the Newton steps) need the
exact inverse of their matrix: a route of an inexact backend is accepted by
[linear_route][diff_soup.nonlinear._kkt.linear_route] (recorded as
``exact=False``) and refused by
[RouteFactor][diff_soup.nonlinear._kkt.RouteFactor].

The small bordered-global (Schur) block is solved with a Tikhonov-regularized
dense solve, [_stable_small_solve][diff_soup.nonlinear._kkt._stable_small_solve].
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup.linear import (
    Analysis,
    Dense,
    FactorContractError,
    SparseTopology,
    analyze,
    empty_factor,
    factor,
    refactor,
    solve,
    solve_checked,
)
from diff_soup.linear._context import DEFAULT_CONTEXT
from diff_soup.linear._factor import check_orientation, effective_transpose, rung_dtype
from diff_soup.linear._primitive import pattern_matvec
from diff_soup.linear._refine import (
    DEFAULT_MAX_STEPS,
    DEFAULT_RESIDUAL_TARGET,
    DIRECT_RESIDUAL_THRESHOLD,
    FIXED_REFINEMENT_STEPS,
    refine,
    refine_fixed,
)
from diff_soup.utils.lanes import lane_cond

#: Analyze options that configure the reduced-precision solve policy rather than
#: the backend: the refinement target and budget of ``"fp32_ir"``, and the
#: acceptance threshold and gate of ``"fp32"``. They are read here and not passed
#: to the backend.
_POLICY_OPTIONS = (
    "ir_target_rel_residual",
    "ir_max_steps",
    "fp32_residual_threshold",
    "fp32_residual_gate",
    "fp32_scaling",
)

_REDUCED_MODES = ("fp32", "fp32_ir")


@dataclasses.dataclass(frozen=True)
class LinearPlan:
    """The pattern and the policies a caller factors a matrix with.

    What a caller declares before any values exist; hashable, so it can be a
    static argument of a compiled program.
    [linear_route][diff_soup.nonlinear._kkt.linear_route] analyses it on the
    contract once the dtype of the values is known.

    Attributes:
        topology: The [SparseTopology][diff_soup.linear.SparseTopology] of
            the matrix.
        layout: The device-placement [Layout][diff_soup.Layout], or ``None``
            for the default placement.
        options: Backend options as ``(key, value)`` pairs.
        precision: The [Precision][diff_soup.Precision] policy (its solve
            mode selects reduced-precision factors), or ``None`` for full
            precision.
    """

    topology: SparseTopology
    layout: Any = None
    options: tuple = ()
    precision: Any = None


@dataclasses.dataclass(frozen=True)
class LinearRoute:
    """How one matrix pattern is factored and solved on the factor contract.

    Hashable, so it can be a static argument of a compiled program.

    Attributes:
        full: The contract analysis at the precision rung of the matrix
            values (``"fp64"`` for ``float64`` values, ``"fp32"`` for
            ``float32``).
        reduced: The ``"fp32"`` analysis of the same pattern when the solve
            mode factors in single precision below double-precision values,
            else ``None``.
        mode: ``"full"``, ``"fp32"`` (a single-precision factor whose
            solution must reduce the relative residual below
            ``direct_threshold``) or ``"fp32_ir"`` (a single-precision factor
            refined to ``residual_target``).
        residual_target: Relative-residual target of ``"fp32_ir"``.
        max_steps: Refinement budget of ``"fp32_ir"``.
        direct_threshold: Acceptance threshold of ``"fp32"``.
        residual_gate: Whether ``"fp32"`` measures the residual of its
            solutions at all (``False`` leaves acceptance to the caller's
            nonlinear iteration).
        adjoint_target: Relative-residual target of the transposed solves of
            [RouteFactor.solve][diff_soup.nonlinear._kkt.RouteFactor.solve],
            or ``None`` to use ``residual_target``. Those are the solves of
            the implicit-function sensitivities' backward pass; the
            reverse-mode derivative of an untransposed solve does not use it.
        exact: Whether the route's backend applies exact inverses
            (``is_exact``). An inexact (iterative) backend's route serves a
            consumer that reads each solve's convergence report;
            [RouteFactor][diff_soup.nonlinear._kkt.RouteFactor], whose
            consumers need an exact inverse, refuses it.
    """

    full: Optional[Analysis]
    reduced: Optional[Analysis] = None
    mode: str = "full"
    residual_target: float = DEFAULT_RESIDUAL_TARGET
    max_steps: int = DEFAULT_MAX_STEPS
    direct_threshold: float = DIRECT_RESIDUAL_THRESHOLD
    residual_gate: bool = True
    adjoint_target: Optional[float] = None
    exact: bool = True


def _rung(dtype: Any) -> str:
    """The contract precision rung that holds values of ``dtype``."""
    dtype = np.dtype(dtype)
    # ``finfo`` of a complex dtype describes its real part.
    width = np.finfo(dtype).bits if np.issubdtype(dtype, np.inexact) else 64
    return "fp32" if width <= 32 else "fp64"


def _context_for(analysis: Analysis):
    """The solve context a factor of ``analysis`` lives in (none for array factors)."""
    if analysis.backend.factor_residency == "arrays":
        return None
    return DEFAULT_CONTEXT


def linear_route(
    backend: Any,
    analysis: Any,
    *,
    dtype: Any,
    topology: Optional[SparseTopology] = None,
) -> LinearRoute:
    """The [LinearRoute][diff_soup.nonlinear._kkt.LinearRoute] of a caller's backend and analysis.

    ``analysis`` may be a contract [Analysis][diff_soup.linear.Analysis], a
    [LinearPlan][diff_soup.nonlinear._kkt.LinearPlan] (its topology, options
    and layout are analysed on the contract, and its
    [Precision][diff_soup.Precision] supplies the solve mode and the
    refinement settings), or ``None`` together with a ``topology``. A
    ``None`` backend is the dense backend. A contract
    analysis at the ``"fp32"`` rung below ``float64`` values refines its
    solutions to double precision. An inexact backend is accepted and
    recorded in the route's ``exact`` field: a consumer that reads every
    solve's convergence report may use it, while
    [RouteFactor][diff_soup.nonlinear._kkt.RouteFactor] refuses it.

    Args:
        backend: The configured [LinearSolver][diff_soup.linear.LinearSolver],
            or ``None`` for [Dense][diff_soup.linear.Dense].
        analysis: A contract analysis, a plan, or ``None``.
        dtype: The dtype of the matrix values the route factors.
        topology: The pattern, required when ``analysis`` is ``None``.

    Returns:
        The route.

    Raises:
        ValueError: If neither ``analysis`` nor ``topology`` gives the pattern.
    """
    rung = _rung(dtype)
    policy = None
    reduced_declared = False
    symmetric = spd = False
    if isinstance(analysis, Analysis):
        backend = analysis.backend
        topology = analysis.topology
        layout = analysis.layout
        options = dict(analysis.options)
        symmetric, spd = analysis.symmetric, analysis.spd
        reduced_declared = analysis.precision == "fp32"
    elif analysis is not None:
        topology = analysis.topology
        layout = getattr(analysis, "layout", None)
        options = dict(getattr(analysis, "options", ()) or ())
        policy = getattr(analysis, "precision", None)
    else:
        if topology is None:
            raise ValueError(
                "linear_route needs an analysis or a topology to know the matrix pattern; "
                "got neither."
            )
        layout = None
        options = {}
    if backend is None:
        backend = Dense()
    exact = bool(backend.is_exact)
    settings = {key: options.pop(key) for key in _POLICY_OPTIONS if key in options}
    if options.get("ir_nsteps") == 0 and type(backend).backend_name != "cudss":
        # No native refinement steps: vacuous on a backend that has none.
        options.pop("ir_nsteps")
    common = dict(
        backend=backend,
        options=options or None,
        layout=layout,
        symmetric=symmetric,
        spd=spd,
    )
    full = analyze(topology, precision=rung, **common)
    mode = getattr(policy, "solve_mode", "fp64")
    if reduced_declared and rung == "fp64":
        mode = "fp32_ir"
    if rung == "fp32" or mode not in _REDUCED_MODES:
        return LinearRoute(full=full, exact=exact)
    adjoint_tol = getattr(policy, "adjoint_ir_tol", None)
    return LinearRoute(
        full=full,
        reduced=analyze(topology, precision="fp32", **common),
        mode=mode,
        residual_target=float(settings.get("ir_target_rel_residual", DEFAULT_RESIDUAL_TARGET)),
        max_steps=int(settings.get("ir_max_steps", DEFAULT_MAX_STEPS)),
        direct_threshold=float(settings.get("fp32_residual_threshold", DIRECT_RESIDUAL_THRESHOLD)),
        residual_gate=str(settings.get("fp32_residual_gate", "auto")) != "outer",
        adjoint_target=None if adjoint_tol is None else float(adjoint_tol),
        exact=exact,
    )


def _reduced_attempt(route: LinearRoute, f: Any, values: Any, b: Any, transpose: bool):
    """``refine`` on a reduced route's single-precision factor ``f``: ``(x, report)``.

    ``"fp32_ir"`` refines to the route's residual target (the adjoint target
    for a transposed solve, when the route has one); ``"fp32"`` measures the
    residual of the factor's own solution against the acceptance threshold.
    """
    if route.mode == "fp32_ir":
        target = route.residual_target
        if transpose and route.adjoint_target is not None:
            target = route.adjoint_target
        steps = route.max_steps
    else:
        target, steps = route.direct_threshold, 0
    x, _, report = refine(
        f, b, values=values, residual_target=target, max_steps=steps, transpose=transpose
    )
    return x, report


def _reduced_steps(route: LinearRoute) -> int:
    """The fixed corrections that refine a reduced route's tangents (none for ``"fp32"``)."""
    return route.max_steps if route.mode == "fp32_ir" else 0


def _full_factor_where(
    route: LinearRoute, values: Any, predicate: Any, orientation: str = "A"
) -> Any:
    """A full-precision factor of ``values`` where ``predicate`` holds, empty elsewhere.

    An empty factor of its own storage refactored with the per-lane
    predicate: no factorization runs when no lane asks, and a lane that does
    not ask keeps an empty factor whose solves are ``nan``. The factor is
    built outside any conditional, so every derivative of a solve with it
    receives its own handle. ``orientation`` is the factor's, ``"A"`` or
    ``"AT"``.
    """
    held = empty_factor(
        route.full, ctx=_context_for(route.full), orientation=orientation, dtype=values.dtype
    )
    return refactor(held, values, predicate)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _fallback_solve(
    route: LinearRoute, transpose: bool, orientation: str, f: Any, values: Any, b: Any
):
    """``(x, stalled)``: a reduced route's solution of ``M x = b``.

    The solution of the single-precision factor ``f`` (of ``orientation``),
    refined or checked against ``values``; where that misses the route's
    target (``stalled``, per lane under ``vmap``), a double-precision
    factorization of ``values`` in the same orientation solves instead. That
    factorization runs only where some lane stalls. The derivative rule
    solves the tangent with the factor that produced ``x``, so reverse mode
    never transposes a solve of the other one.
    """
    x, report = _reduced_attempt(route, f, values, b, transpose)
    full = _full_factor_where(route, values, report.stalled, orientation)
    x_full, _ = solve(full, b, transpose=transpose)
    return jnp.where(report.stalled, x_full.astype(x.dtype), x), report.stalled


def _tangent_solve(fac: Any, keep: Any, x: Any, dvalues: Any, db: Any, transpose: bool):
    """``solve(fac, db - dM x)`` where ``keep`` holds, zero elsewhere.

    The tangent of the solution ``x`` of ``M x = b`` along the value tangent
    ``dvalues`` and the right-hand-side tangent ``db``, with ``M`` the
    operator ``fac`` holds. The right-hand side and the result are both
    masked by ``keep``, so a lane whose factor failed or is empty never
    passes its ``nan`` solve on, in either direction.
    """
    effective = effective_transpose(fac.orientation, transpose)
    rhs = db - pattern_matvec(fac.analysis, dvalues, x, transpose=effective)
    rhs = jnp.where(keep, rhs, jnp.zeros_like(rhs))
    dx, _ = solve(fac, rhs, transpose=transpose)
    dx = dx.astype(x.dtype)
    return jnp.where(keep, dx, jnp.zeros_like(dx))


def _refined_tangent(
    route: LinearRoute, f: Any, values: Any, keep: Any, x: Any, dvalues: Any, db: Any, transpose
):
    """The reduced factor's tangent ``A^-1 (db - dA x)``, refined, where ``keep`` holds.

    The single-precision factor's solve followed by the route's fixed
    corrections against ``values`` (``max_steps`` for ``"fp32_ir"``, none
    for ``"fp32"``), the tangent rule of
    [refine][diff_soup.linear.refine]; masked by ``keep`` as
    [_tangent_solve][diff_soup.nonlinear._kkt._tangent_solve] is.
    """
    effective = effective_transpose(f.orientation, transpose)
    rhs = db - pattern_matvec(f.analysis, dvalues, x, transpose=effective)
    rhs = jnp.where(keep, rhs, jnp.zeros_like(rhs))
    dx, _ = refine_fixed(f, rhs, values=values, steps=_reduced_steps(route), transpose=transpose)
    dx = dx.astype(x.dtype)
    return jnp.where(keep, dx, jnp.zeros_like(dx))


@_fallback_solve.defjvp
def _fallback_solve_jvp(route, transpose, orientation, primals, tangents):
    """Tangent of ``_fallback_solve``, solved by the factor that produced the solution.

    The primal comes from ``_fallback_solve`` itself, so a further derivative
    of it uses this rule again. Where the route stalled the tangent factors
    ``values`` at double precision once more (no factorization where no lane
    stalled); elsewhere it is the single-precision factor's solve refined by
    the route's fixed corrections. Both factors are built outside any
    conditional and the two tangents are selected per lane.
    """
    f, values, b = primals
    _, dvalues, db = tangents
    x, stalled = _fallback_solve(route, transpose, orientation, f, values, b)
    full = _full_factor_where(route, values, stalled, orientation)
    dx_full = _tangent_solve(full, stalled, x, dvalues, db, transpose)
    kept = jnp.logical_not(stalled)
    dx_reduced = _refined_tangent(route, f, values, kept, x, dvalues, db, transpose)
    dx = jnp.where(stalled, dx_full, dx_reduced)
    return (x, stalled), (dx, np.zeros(np.shape(stalled), dtype=jax.dtypes.float0))


class RouteFactor:
    """One factorization of a matrix on a route, with its solves.

    The matrix is factored once, at construction, in the route's working
    precision and in the orientation the caller names: ``"A"`` factors the
    matrix ``M = A``, ``"AT"`` factors its transpose ``M = A^T`` directly (a
    reverse sweep that solves only transposed systems factors ``"AT"`` and
    never builds a factor of ``A``). Every solve, with either operator
    ``M`` or ``M^T`` and with any number of right-hand sides, reuses that
    factorization. Derivatives with respect to the values flow through the
    solves by the factor contract's rules.

    The consumers of a route factor (the implicit-function derivatives, the
    consistent initialisation, the projections and the Newton steps of the
    nonlinear solvers) treat every solve as an exact inverse, so a route of
    an inexact backend is refused here.

    The solve methods return the solution only. A route factor is factored
    once and never refactored or invalidated, so no operation has to wait
    for its solves: each of them follows the factorization through the
    factor's handle, and solves on one factor may run in any order.

    Attributes:
        route: The [LinearRoute][diff_soup.nonlinear._kkt.LinearRoute].
        values: The matrix values on the route's pattern.
        orientation: ``"A"`` or ``"AT"``, the operator ``M`` the factor holds.
        factor: The contract [Factor][diff_soup.linear.Factor] (single
            precision on a reduced route).
    """

    def __init__(self, route: LinearRoute, values: Any, *, orientation: str = "A"):
        """Factor ``values`` on ``route``.

        Args:
            route: The route of the pattern.
            values: Matrix values on the route's pattern.
            orientation: ``"A"`` (the default) factors the matrix of
                ``values``; ``"AT"`` factors its transpose directly.

        Raises:
            FactorContractError: If the route's backend is inexact
                (``is_exact=False``), or the orientation is not ``"A"`` or
                ``"AT"``.
        """
        orientation = check_orientation(orientation)
        if not route.exact:
            name = type(route.full.backend).__name__
            raise FactorContractError(
                f"The implicit-function, consistent-initialisation, projection and Newton-step "
                f"solves need the exact inverse of their matrix, but the backend {name} is "
                "inexact (is_exact=False): its solves stop at a tolerance. Use an exact backend "
                "for these solves (Dense, KLU, SuperLU or CuDSS)."
            )
        values = jnp.asarray(values)
        self.route = route
        self.values = values
        self.orientation = orientation
        self.n = int(route.full.n)
        working = route.full if route.reduced is None else route.reduced
        held = values if route.reduced is None else values.astype(rung_dtype("fp32", values.dtype))
        self.factor = factor(working, held, ctx=_context_for(working), orientation=orientation)

    def solve(self, b: Any, *, transpose: bool = False) -> jax.Array:
        """Solve ``M x = b`` (or ``M^T x = b``) to the route's accuracy.

        ``M`` is the operator the factor holds (the matrix for orientation
        ``"A"``, its transpose for ``"AT"``), as on the factor contract.

        A full-precision route solves with its factor. A reduced route
        refines the single-precision solution (``"fp32_ir"``) or checks its
        residual (``"fp32"``) against the double-precision values, and a
        solution that misses its target is replaced by a double-precision
        factorization and solve. Not linear in ``b`` on a reduced route:
        use [linear_solve][diff_soup.nonlinear._kkt.RouteFactor.linear_solve]
        where ``b`` is a tangent inside a custom differentiation rule.

        On a reduced route the derivative is a solve with the factor that
        produced the solution, at that solution: the single-precision
        factor's where the target was met, the double-precision factor's
        where the fallback ran. Reverse mode runs only that factor's
        transposed solve, so a single-precision factor that failed (and
        solves to ``nan``) never enters the derivative.

        Args:
            b: Right-hand side, ``(n,)`` or ``(nrhs, n)``.
            transpose: Solve with ``M^T``.

        Returns:
            The solution, with the shape of ``b``.
        """
        route = self.route
        if route.reduced is None:
            return solve(self.factor, b, transpose=transpose)[0]
        if route.mode == "fp32" and not route.residual_gate:
            return refine_fixed(self.factor, b, values=self.values, steps=0, transpose=transpose)[0]
        x, _ = _fallback_solve(
            route, bool(transpose), self.orientation, self.factor, self.values, jnp.asarray(b)
        )
        return x

    def linear_solve(self, b: Any, *, transpose: bool = False) -> jax.Array:
        """Solve ``M x = b`` (or ``M^T x = b``) with a function linear in ``b``.

        For a right-hand side that is a tangent inside a custom
        differentiation rule, which JAX transposes and differentiates
        again. A full-precision route solves with its factor; a reduced
        route applies a fixed number of refinement corrections (none for
        ``"fp32"``) instead of the adaptive sweep of
        [solve][diff_soup.nonlinear._kkt.RouteFactor.solve]; the corrections
        refine the derivative with respect to the values as well (see
        [refine_fixed][diff_soup.linear._refine.refine_fixed]).

        Where the single-precision factorization failed (its status, read by
        a checked solve of a constant right-hand side, is not ``SUCCESS``;
        per lane under ``vmap``), the solution comes from a double-precision
        factorization and solve instead, which runs only when some lane
        failed; its derivatives are that factor's. The choice depends on the
        matrix alone, so the result stays linear in ``b``.

        Args:
            b: Right-hand side, ``(n,)`` or ``(nrhs, n)``.
            transpose: Solve with ``M^T``.

        Returns:
            The solution, with the shape of ``b``.
        """
        route = self.route
        if route.reduced is None:
            return solve(self.factor, b, transpose=transpose)[0]
        steps = FIXED_REFINEMENT_STEPS if route.mode == "fp32_ir" else 0
        b = jnp.asarray(b)
        failed = jnp.logical_not(self.solves())

        def zero(x: Any) -> Any:
            return jnp.zeros_like(x)

        def served(x: Any) -> Any:
            """``x`` where the reduced factor serves the lane, a constant copy where it failed."""
            return jnp.where(failed, jax.lax.stop_gradient(x), x)

        # Each branch sees the right-hand side only where its factor serves the
        # lane and returns zero elsewhere, and on a failed lane the reduced
        # branch's values carry no derivative (its iterates are nan there), so a
        # failed or unused factor never passes its nan solve on in either
        # direction.
        held = eqx.tree_at(lambda fac: fac.values, self.factor, served(self.factor.values))
        reduced, _ = refine_fixed(
            held,
            jnp.where(failed, zero(b), b),
            values=served(self.values),
            steps=steps,
            transpose=transpose,
        )
        reduced = jnp.where(failed, zero(reduced), reduced)

        def full_precision(operands: tuple) -> Any:
            values, rhs = operands
            full = factor(
                route.full, values, ctx=_context_for(route.full), orientation=self.orientation
            )
            x, _ = solve(full, jnp.where(failed, rhs, zero(rhs)), transpose=transpose)
            x = x.astype(reduced.dtype)
            return jnp.where(failed, x, zero(x))

        fallback = lane_cond(
            failed,
            full_precision,
            lambda operands: zero(reduced),
            (self.values, b),
            false_is_free=True,
        )
        return jnp.where(failed, fallback, reduced)

    def solves(self) -> jax.Array:
        """Boolean scalar: whether the factorization solves (a primal-only test).

        It is the status of a checked solve of a constant right-hand side.
        """
        dtype = self.values.dtype
        _, _, report = solve_checked(self.factor, jnp.ones((self.n,), dtype=dtype))
        return report.ok


def _stable_small_solve(mat: jax.Array, rhs: jax.Array, eps: float) -> jax.Array:
    """Regularized small-block solve for Schur complement systems.

    Uses LU-based ``jnp.linalg.solve`` with Tikhonov regularisation (``eps * I``)
    instead of the slower SVD-based ``pinv``. The ``eps * I`` term is applied
    **only here** -- callers pass the raw Schur complement without extra
    regularisation.

    Args:
        mat: Schur-complement matrix, shape ``(m, m)``.
        rhs: Right-hand side, shape ``(m,)``.
        eps: Tikhonov regularisation magnitude.

    Returns:
        The regularised solution, shape ``(m,)`` (zeros when ``m == 0``).
    """
    m = mat.shape[0]
    if m == 0:
        return jnp.zeros_like(rhs)
    eye = jnp.eye(m, dtype=mat.dtype)
    reg = jnp.asarray(eps, dtype=mat.dtype)
    mat_reg = mat + reg * eye
    return jnp.linalg.solve(mat_reg, rhs)


def _cast_like(x: jax.Array, like: jax.Array) -> jax.Array:
    """``x`` in the dtype of ``like`` (a solve returns its precision rung's dtype)."""
    like = jnp.asarray(like)
    return x if x.dtype == like.dtype else x.astype(like.dtype)


def transposable_solve(
    route: LinearRoute,
    values: Any,
    rhs: Any,
    *,
    reverse_route: Optional[LinearRoute] = None,
    adjoint_dtype: Any = None,
    transpose_solve: Any = None,
) -> jax.Array:
    """``x = A^-1 rhs``, linear in ``rhs``, whose transpose factors ``A^T`` directly.

    The solve of the implicit-function derivative rules. ``A`` is the matrix
    of ``values`` on ``route``'s pattern. Evaluated (a forward tangent), it
    factors ``A`` on ``route`` and solves with that factor, linearly in
    ``rhs`` as [RouteFactor.linear_solve][diff_soup.nonlinear._kkt.RouteFactor.linear_solve]
    does. Transposed by JAX (the cotangent solve of a reverse-mode gradient),
    it factors ``A^T`` directly, in orientation ``"AT"`` on ``reverse_route``
    (``route`` by default) and at ``adjoint_dtype`` when one is given (the
    values are cast up before factoring and the solution cast back to the
    cotangent's dtype), and solves with that factor: a reverse sweep that
    solves only transposed systems never builds a factor of ``A`` and never
    asks a backend for the transposed solve of one. ``transpose_solve``, a
    function of the cotangent, replaces that transposed solve entirely (an
    iterative backend's preconditioned transposed solve, say).

    Differentiated again, in either mode, the derivative of the solution
    along a tangent ``dA`` of the values is ``A^-1 (drhs - dA x)`` (or its
    transpose), with ``dA x`` the pattern matrix-vector product of the
    contract; so a solve inside a ``custom_jvp`` rule supports
    reverse-over-forward, forward-over-reverse and ``jax.hessian`` without
    any entry chosen at trace time. The construction is
    ``jax.lax.custom_linear_solve``.

    ``values`` may also be a function of no arguments that returns the
    values. The function is then evaluated inside the solves themselves, so
    a matrix that only a transposed solve needs is assembled only when that
    solve runs. Under ``jax.vjp`` the values then belong to the cotangent
    pass instead of the forward pass: a reverse sweep that linearizes a step
    without applying its pullback (the recomputed steps of a checkpointed
    loop) assembles no matrix for it.

    Args:
        route: The route of the pattern, for the forward solve.
        values: Matrix values on the route's pattern, or a function of no
            arguments returning them.
        rhs: Right-hand side, ``(n,)`` or ``(nrhs, n)``.
        reverse_route: The route of the transposed factor, when the reverse
            sweep declares its own precision policy; ``None`` uses ``route``.
        adjoint_dtype: The dtype the transposed factorization and solve run
            at, or ``None`` for the values' own.
        transpose_solve: ``cotangent -> solution`` replacing the transposed
            factor's solve, or ``None``.

    Returns:
        The solution, with the shape and dtype of ``rhs``.
    """
    if callable(values):
        values_of = values
    else:
        values = jnp.asarray(values)

        def values_of():
            """The values the caller passed."""
            return values

    rhs = jnp.asarray(rhs)
    reverse = route if reverse_route is None else reverse_route
    analysis = route.full

    def matvec(x):
        """``A x`` on the pattern: read for its derivative along the values only."""
        return pattern_matvec(analysis, values_of(), x, transpose=False).astype(rhs.dtype)

    def solve(_, b):
        """The forward solve: a factor of ``A`` on ``route``."""
        held = RouteFactor(route, jnp.asarray(values_of()), orientation="A")
        return held.linear_solve(b).astype(b.dtype)

    def solve_transposed(_, ct):
        """The reverse solve: a factor of ``A^T`` on ``reverse``, at ``adjoint_dtype``."""
        if transpose_solve is not None:
            return jnp.asarray(transpose_solve(ct)).astype(ct.dtype)
        values = jnp.asarray(values_of())
        held_values = values if adjoint_dtype is None else values.astype(adjoint_dtype)
        held = RouteFactor(reverse, held_values, orientation="AT")
        return held.linear_solve(ct.astype(held_values.dtype)).astype(ct.dtype)

    return jax.lax.custom_linear_solve(matvec, rhs, solve, solve_transposed, symmetric=False)


def kkt_solve(rf, B, C, D, rhs_u, rhs_lam, schur_eps, *, linear=False):
    """Solve ``[[A, B], [C, D]] [du; dlam] = [rhs_u; rhs_lam]`` given ``A``'s factor.

    ``rf`` is the [RouteFactor][diff_soup.nonlinear._kkt.RouteFactor] of the
    state block ``A``, in either orientation (an ``"AT"`` factor serves the
    solves with ``A`` as its transposed solves); ``B``, ``C``, ``D`` are the
    coupling and bordered-global blocks. Eliminates the state block, forms
    the Schur complement ``D - C @ du_B``, and back-substitutes. The driving
    right-hand side and the coupling block are solved separately: packing
    them in one array mixes tangent-dependent and primal-only slices, which
    makes a ``custom_jvp`` using this solve impossible for JAX to transpose.
    A border-free system (``D`` empty) short-circuits to a single state
    solve. With ``linear`` the state solves are linear in their right-hand
    sides (see
    [RouteFactor.linear_solve][diff_soup.nonlinear._kkt.RouteFactor.linear_solve]),
    for a tangent right-hand side inside a custom differentiation rule.
    """
    transpose = rf.orientation == "AT"
    if linear:

        def state_solve(b):
            return rf.linear_solve(b, transpose=transpose)

    else:

        def state_solve(b):
            return rf.solve(b, transpose=transpose)

    m = int(D.shape[0])
    if m == 0:
        return _cast_like(state_solve(rhs_u), rhs_u), jnp.zeros_like(rhs_lam)
    du_r = _cast_like(state_solve(rhs_u), rhs_u)
    du_B = _cast_like(state_solve(B.T), B).T
    schur = D - C @ du_B
    dlam = _stable_small_solve(schur, rhs_lam - C @ du_r, schur_eps)
    du = du_r - du_B @ dlam
    return du, dlam


def kkt_solve_transposed(rf, B, C, D, ct_u, ct_lam, schur_eps):
    """Solve the transposed system ``[[A.T, C.T], [B.T, D.T]]`` given ``A``'s factor.

    The adjoint of [kkt_solve][diff_soup.nonlinear._kkt.kkt_solve]: the state
    solves are solves with ``A^T``, transposed solves of a factor of
    orientation ``"A"`` and plain solves of one of orientation ``"AT"`` (a
    caller that solves only this system factors ``"AT"`` and never builds a
    factor of ``A``), and the coupling rows fed to the multi-solve are ``C``
    (whose rows are the columns of ``C.T``). Same elimination shape,
    transposed throughout.
    """
    transpose = rf.orientation == "A"
    m = int(D.shape[0])
    if m == 0:
        return _cast_like(rf.solve(ct_u, transpose=transpose), ct_u), jnp.zeros_like(ct_lam)
    sol = rf.solve(jnp.concatenate((ct_u[None, :], C), axis=0), transpose=transpose)
    sol = _cast_like(sol, ct_u)
    x_ct = sol[0]
    x_Ct = sol[1:].T
    schur_t = D.T - B.T @ x_Ct
    q = _stable_small_solve(schur_t, ct_lam - B.T @ x_ct, schur_eps)
    p = x_ct - x_Ct @ q
    return p, q
