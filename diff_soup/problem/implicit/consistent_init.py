"""Consistent initialization on the full implicit residual, with exact derivatives.

Given pinned differential values ``y_d`` at ``t0``, the solver finds the
algebraic values ``y_a`` and differential slopes ``ydot_d`` — with ``ydot_a``
structurally zero — as the joint root of the full residual
``F(t0, y, ydot, theta, control) = 0`` in the unknowns ``(y_a, ydot_d)``.
The joint unknown vector is laid out state-aligned: slot ``s`` holds ``y[s]``
when variable ``s`` is algebraic and ``ydot[s]`` when it is differential, so
the exact Jacobian in the unknowns is the column blend

    K[:, s] = F_y[:, s]  (s algebraic)   |   K[:, s] = F_ydot[:, s]  (s differential),

which drives both the Newton iteration and the implicit-function derivatives
at the converged root. The state-aligned layout keeps the residual's own
structural diagonal on ``K``'s diagonal, which static-pivoting sparse
factorizations rely on for accuracy. Working on the full residual handles
derivative couplings in any equation row. Supplied algebraic values and
slope seeds are initial guesses only: they may accelerate the solve but
never define it, and they carry no derivative. Pinned differential values,
``theta``, and ``control`` differentiate through the full residual.

``K`` is solved through a linear backend selected by size and platform (the
``backend`` argument): tiny systems use a dense LU, larger ones keep ``K``
as sparse values over its static column-selected topology and factorize
through a sparse-direct backend (KLU on CPU, cuDSS on GPU) in both the
primal orientation and, for a reverse sweep, the transposed one — the sparse
path never materializes an ``(n, n)`` matrix. The resolved route is recorded
on the plan as ``backend_name``.

Each route has one derivative entry, a ``custom_jvp``
([consistent_ic_root][diff_soup.problem.implicit.consistent_init.consistent_ic_root]
for the joint solve, its reduced twin for the algebraic-block elimination).
Its rule is linear in the tangents, so JAX derives reverse mode by
transposing it: the cotangent solve factors the transposed matrix directly,
on the plan's reverse precision policy. The rule computes its primal through
the entry itself, so a further differentiation (``jax.hessian``, reverse over
forward) differentiates the root by the same rule, never through the Newton
loop. A solve that fails to converge — or whose sensitivity system is
singular — has no supported derivative: its tangents are NaN, its
reverse-mode cotangents are exact zeros (the rule zeroes the tangents of an
invalid root, and a transposition cannot tell a consumed cotangent from a
discarded one), and the returned diagnostics record the failure. [check_consistent_init][diff_soup.problem.implicit.consistent_init.check_consistent_init] is the eager companion
that explains such failures, naming the implicated unknowns and their
declared partitions when the joint system is singular.
"""

from __future__ import annotations

import functools
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import zero_from_primal

from diff_soup.linear import LinearSolver, SparseTopology
from diff_soup.linear._resolver import BackendResolver
from diff_soup.nonlinear._kkt import LinearPlan, RouteFactor, linear_route, transposable_solve
from diff_soup.problem.implicit.linearize import build_linearization_plan

__all__ = [
    "ConsistentInitPlan",
    "ConsistentInitResult",
    "build_consistent_init_plan",
    "check_consistent_init",
    "consistent_ic_root",
    "consistent_ic_root_reduced",
    "consistent_initial_state",
    "reduced_route_eligible",
]


def reduced_route_eligible(problem) -> bool:
    """Whether the eliminated (reduced) initialization route serves ``problem``.

    The reduced route is a mathematical capability of the diagonal invertible
    mass form only: the problem is a
    [MassProblem][diff_soup.problem.implicit.problems.MassProblem] whose
    deduplicated derivative-coefficient support holds exactly one ``(d, d)``
    entry for every differential slot and nothing else. With that structure
    the joint initialization matrix is block-triangular, so the algebraic
    values solve on the algebraic block alone and the differential slopes
    reconstruct as ``ydot_d = f_d / D_dd``. Every fully implicit problem and
    every cross-row or non-diagonal operator takes the full joint solve. The
    decision reads the mass operator's declared structure only — never a
    source's authoring surface. Invertibility of the diagonal is a value
    property checked at the solved point, not here.
    """
    from diff_soup.problem.implicit.problems import MassProblem

    if not isinstance(problem, MassProblem):
        return False
    structure = problem.structure
    n = int(structure.n)
    topology = problem.mass.coefficient_sparsity
    rows = np.asarray(topology.rows, dtype=np.int64)
    cols = np.asarray(topology.cols, dtype=np.int64)
    support = np.unique(rows * n + cols)
    d_idx = np.asarray(structure.differential_indices, dtype=np.int64)
    diagonal = np.unique(d_idx * n + d_idx)
    return support.shape == diagonal.shape and bool(np.all(support == diagonal))


#: Process-global backend singletons for name-based selection (one KLU / SuperLU /
#: cuDSS instance per process, mirroring the engines' resolution semantics).
_BACKEND_SINGLETONS: dict = {}


def _backend_instance(name: str) -> LinearSolver:
    """Return the process-global backend instance for a concrete sparse name."""
    instance = _BACKEND_SINGLETONS.get(name)
    if instance is not None:
        return instance
    if name == "klu":
        from diff_soup.linear import KLU

        instance = KLU()
    elif name == "superlu":
        from diff_soup.linear import SuperLU
        from diff_soup.linear.superlu_backend import require_superlu

        require_superlu("backend='superlu'")
        instance = SuperLU()
    elif name == "cudss":
        from diff_soup.linear import CuDSS

        if CuDSS is None:
            raise RuntimeError(
                "the cuDSS backend was requested for consistent initialization "
                "but its native extension is not available on this host; "
                "install the cuDSS build, or select backend='klu' (CPU) or "
                "backend='dense'."
            )
        instance = CuDSS()
    else:
        raise ValueError(
            f"no sparse linear backend is named {name!r}; expected 'klu', 'superlu' or 'cudss'."
        )
    _BACKEND_SINGLETONS[name] = instance
    return instance


def resolve_backend_route(backend, *, plan_mode: str, n_dofs: int):
    """Resolve a backend request to its concrete route ``(name, instance)``.

    ``backend`` is ``"auto"``, a concrete name (``"dense"`` / ``"klu"`` /
    ``"superlu"`` / ``"cudss"``), or a
    [LinearSolver][diff_soup.linear.LinearSolver] instance.
    ``"auto"`` stays dense for a dense-mode linearization plan (the joint
    matrix is assembled densely regardless) and otherwise picks the
    platform's route from the system size. The returned ``instance`` is
    ``None`` on the dense route. Two requests that resolve to the same name
    are interchangeable for the joint initialization solve: the backend
    instances hold no per-solve state, and their symbolic analyses are
    memoised process-globally on topology content. An iterative backend
    ([Krylov][diff_soup.linear.Krylov]) resolves to its exact inner backend:
    the initialization's Newton steps and its implicit-function derivative
    solve with exact inverses.
    """
    if isinstance(backend, LinearSolver):
        from diff_soup.linear._krylov import exact_backend

        backend = exact_backend(backend)
        name = backend.backend_name
        return name, (None if name == "dense" else backend)
    if isinstance(backend, str):
        if backend == "auto" and plan_mode == "dense":
            return "dense", None
        platform = "gpu" if jax.default_backend() == "gpu" else "cpu"
        name = BackendResolver(backend=backend).choose(block_size=n_dofs, platform=platform)
        return name, (None if name == "dense" else _backend_instance(name))
    raise ValueError(
        f"backend must be 'auto', 'dense', 'klu', 'superlu', 'cudss', or a "
        f"LinearSolver instance; got {backend!r}."
    )


class ConsistentInitResult(eqx.Module):
    """The result of a consistent initialization.

    Attributes:
        y: The consistent state.
        ydot: The consistent derivative, zero on algebraic rows.
        residual_norm: The residual 2-norm at `y` and `ydot`.
        iterations: The Newton iterations used.
        converged: Whether `residual_norm` reached `tol`.
    """

    y: jax.Array
    ydot: jax.Array
    residual_norm: jax.Array
    iterations: jax.Array
    converged: jax.Array


class ConsistentInitPlan:
    """Static machinery for one problem's joint consistent-initialization solve.

    Identity-hashed (like the linearization plan it wraps) so it can ride a
    compile cache as a static value. Build with
    [build_consistent_init_plan][diff_soup.problem.implicit.consistent_init.build_consistent_init_plan].

    Attributes:
        route: The solve the plan runs — ``"joint"`` (the full state-aligned
            system) or ``"reduced"`` (algebraic-block elimination with slope
            reconstruction, available only for a diagonal invertible mass;
            see [reduced_route_eligible][diff_soup.problem.implicit.consistent_init.reduced_route_eligible]).
        backend_name: The resolved linear route — ``"dense"``, ``"klu"``,
            ``"superlu"`` or ``"cudss"``. Diagnostics read this to see which solver actually
            serves the solve (including the dense fallback that ``"auto"``
            selects for tiny systems).
        k_topology: The static COO pattern of ``K`` over the selected columns
            (sparse-mode joint plans only, else ``None``).
        reduced_topology: The compacted ``(n_a, n_a)`` COO pattern of the
            algebraic block (sparse-mode reduced plans with algebraic
            unknowns, else ``None``).
        linear_backend: The sparse backend instance serving the solves, or
            ``None`` on the dense route (which factors the dense matrix with
            [Dense][diff_soup.linear.Dense]).
        linear_analysis: The [LinearPlan][diff_soup.nonlinear._kkt.LinearPlan]
            (or a caller-supplied analysis) of the factored pattern, or
            ``None`` on the dense route. The Newton solves and the forward
            tangent solve run on it.
        reverse_linear_analysis: The analysis the transposed factor of a
            reverse sweep runs on: a ``LinearPlan`` with
            ``reverse_precision`` when one is declared, else
            ``linear_analysis``.
        layout: The device-placement policy the analysis was built with, or
            ``None`` for the backend's default placement.
        precision: The dtype policy the analysis was built with, or ``None``
            for the backend's default (fp64).
        reverse_precision: The dtype policy of the transposed solves, or
            ``None`` to use ``precision``.
        adjoint_dtype: The ``adjoint_compute_dtype`` of the reverse policy
            (``reverse_precision``, else ``precision``), or ``None``.
    """

    __hash__ = object.__hash__
    __eq__ = object.__eq__

    def __init__(
        self,
        problem,
        plan=None,
        *,
        tol: float = 1e-12,
        max_iters: int = 30,
        chord_iters: int = 8,
        backend="auto",
        analysis=None,
        route: str = "auto",
        layout=None,
        precision=None,
        reverse_precision=None,
    ):
        """Precompute the index maps, the exact-Jacobian assembly, and the route.

        Args:
            problem: The solver problem.
            plan: Optional prebuilt linearization plan for ``problem``.
            tol: Absolute residual 2-norm gate for convergence.
            max_iters: Total Newton iteration budget (chord and exact phases
                together).
            chord_iters: Iteration budget for the frozen-factor (chord)
                phase that opens the solve. The chord phase reuses one
                iteration matrix factored at the seed; poor contraction, a
                nonfinite update or residual, or an exhausted chord budget
                hands the remaining budget to fresh-Jacobian exact Newton.
                ``0`` disables the chord phase (pure exact Newton).
            backend: Linear-solver selection for the factored system:
                ``"auto"`` (dense for tiny systems, else the platform's
                sparse-direct backend), a concrete name (``"dense"`` /
                ``"klu"`` / ``"superlu"`` / ``"cudss"``), or a
                [LinearSolver][diff_soup.linear.LinearSolver] instance.
            analysis: Optional prebuilt symbolic analysis of the factored
                pattern for the selected backend (skips re-analysis).
            layout: Device-placement policy for the factored system's
                symbolic analysis. ``None`` (the default) leaves the
                backend's own default placement. Pass the layout the problem
                was lowered with so the initialization factor lands where
                every other factor of the solve does.
            precision: Dtype policy for the factored system's symbolic
                analysis. ``None`` (the default) leaves the backend's own
                default (fp64). It serves the Newton solves and the forward
                tangent solve of the implicit-function derivative.
            reverse_precision: Dtype policy for the transposed solve of a
                reverse sweep: the transposed factor is analysed with it, and
                its ``adjoint_compute_dtype`` (when declared) sets the dtype
                that solve runs at. ``None`` (the default) uses
                ``precision``.
            route: ``"auto"`` (the default) selects the reduced
                algebraic-block elimination exactly when the problem is
                eligible and the full joint solve otherwise; ``"joint"``
                forces the full state-aligned solve; ``"reduced"`` forces
                the elimination (rejected when the problem's derivative
                operator is not a diagonal mass).

        Raises:
            ValueError: If ``route`` is not one of the three names, or
                ``"reduced"`` is requested for a problem without a diagonal
                mass operator.
        """
        self.plan = build_linearization_plan(problem) if plan is None else plan
        structure = problem.structure
        self.n = int(structure.n)
        self.diff_mask = structure.differential_vars
        self.a_idx = structure.algebraic_indices
        self.d_idx = structure.differential_indices
        self.n_a = int(self.a_idx.size)
        self.tol = float(tol)
        self.max_iters = int(max_iters)
        self.chord_iters = int(chord_iters)
        self.layout = layout
        self.precision = precision
        self.reverse_precision = reverse_precision
        policy = precision if reverse_precision is None else reverse_precision
        self.adjoint_dtype = getattr(policy, "adjoint_compute_dtype", None)
        self.structure = structure

        if route not in ("joint", "reduced", "auto"):
            raise ValueError(f"route must be 'joint', 'reduced', or 'auto'; got {route!r}.")
        if route == "auto":
            route = "reduced" if reduced_route_eligible(problem) else "joint"
        elif route == "reduced" and not reduced_route_eligible(problem):
            raise ValueError(
                "the reduced initialization route requires a diagonal mass: a "
                "MassProblem whose deduplicated derivative support is exactly "
                "one (d, d) entry per differential slot. This problem's "
                "derivative structure does not satisfy that; use route='joint' "
                "(or 'auto')."
            )
        self.route = route

        # The joint unknowns are state-aligned: slot s holds y[s] when
        # variable s is algebraic and ydot[s] when it is differential, so
        # K's columns stay in the state's own column coordinates and the
        # residual's structural diagonal lands on K's diagonal.
        self.k_topology = None
        self.reduced_topology = None
        if route == "reduced":
            self._derivative_apply = problem.mass.apply
            self._build_reduced_maps()
        elif self.plan.mode == "sparse":
            fy_rows = np.asarray(self.plan.fy_topology.rows)
            fy_cols = np.asarray(self.plan.fy_topology.cols)
            keep = ~self.diff_mask[fy_cols]
            self._fy_keep = np.flatnonzero(keep)
            self._fy_rows = fy_rows[keep]
            self._fy_cols = fy_cols[keep]

            self._fyd_rows = np.asarray(self.plan.fydot_topology.rows)
            self._fyd_cols = np.asarray(self.plan.fydot_topology.cols)

            self.k_topology = SparseTopology(
                rows=np.concatenate([self._fy_rows, self._fyd_rows]),
                cols=np.concatenate([self._fy_cols, self._fyd_cols]),
                n_dofs=self.n,
            )

        self._resolve_linear_route(backend, analysis)

    def _build_reduced_maps(self) -> None:
        """Precompute the algebraic-block gather maps for the reduced route.

        The reduced system's unknowns are the local- and global-algebraic slots,
        its equations the algebraic residual rows, and its exact
        Jacobian the ``F_y`` restriction to those rows and columns. Sparse
        plans compact that restriction to an ``(n_a, n_a)`` pattern; dense
        plans slice the dense ``F_y``.
        """
        if self.n_a == 0 or self.plan.mode == "dense":
            return
        to_compact = np.full((self.n,), -1, dtype=np.int64)
        to_compact[np.asarray(self.a_idx, dtype=np.int64)] = np.arange(self.n_a)
        fy_rows = np.asarray(self.plan.fy_topology.rows, dtype=np.int64)
        fy_cols = np.asarray(self.plan.fy_topology.cols, dtype=np.int64)
        keep = (~self.diff_mask[fy_rows]) & (~self.diff_mask[fy_cols])
        self._ra_keep = np.flatnonzero(keep)
        self._ra_rows = to_compact[fy_rows[keep]].astype(np.int32)
        self._ra_cols = to_compact[fy_cols[keep]].astype(np.int32)
        self.reduced_topology = SparseTopology(
            rows=self._ra_rows, cols=self._ra_cols, n_dofs=self.n_a
        )

    def _resolve_linear_route(self, backend, analysis) -> None:
        """Resolve the requested backend to a concrete route and analysis.

        A declared ``layout`` / ``precision`` is recorded in the plan's
        [LinearPlan][diff_soup.nonlinear._kkt.LinearPlan], so the
        initialization factor is placed and typed by the same policy as the
        rest of the solve; either left ``None`` keeps the default for that
        policy. The pattern is analysed on the factor contract when the first
        matrix is factored.
        """
        block = self.n_a if self.route == "reduced" else self.n
        if self.route == "reduced" and self.n_a == 0:
            # No algebraic unknowns: the reduced route is pure slope
            # reconstruction with no linear system at all.
            self.backend_name = "dense"
            self.linear_backend = None
            self.linear_analysis = None
            self.reverse_linear_analysis = None
            return
        name, instance = resolve_backend_route(backend, plan_mode=self.plan.mode, n_dofs=block)
        self.backend_name = name
        if name == "dense" or instance is None:
            self.linear_backend = None
            self.linear_analysis = None
            self.reverse_linear_analysis = None
            return
        topology = self.k_topology if self.route == "joint" else self.reduced_topology
        if self.plan.mode != "sparse":
            topology = SparseTopology.full(block)
        self.linear_backend = instance
        self.linear_analysis = (
            analysis
            if analysis is not None
            else LinearPlan(topology=topology, layout=self.layout, precision=self.precision)
        )
        self.reverse_linear_analysis = (
            self.linear_analysis
            if self.reverse_precision is None
            else LinearPlan(topology=topology, layout=self.layout, precision=self.reverse_precision)
        )


def _plan_root_entry(icp: "ConsistentInitPlan"):
    """The plan's compiled root-solve entry.

    Dispatches on the plan's route (joint or reduced). Jitted once per plan
    and stored on it, so eager repeat calls replay one compiled program
    instead of re-tracing the Newton solve; inside an outer trace the jit
    inlines transparently. Living on the plan, the compiled entry shares the
    plan's own lifetime.
    """
    entry = getattr(icp, "_root_entry", None)
    if entry is None:
        root = consistent_ic_root_reduced if icp.route == "reduced" else consistent_ic_root
        entry = jax.jit(functools.partial(root, icp))
        icp._root_entry = entry
    return entry


def build_consistent_init_plan(
    problem,
    plan=None,
    *,
    tol: float = 1e-12,
    max_iters: int = 30,
    chord_iters: int = 8,
    backend="auto",
    analysis=None,
    route: str = "auto",
    layout=None,
    precision=None,
    reverse_precision=None,
):
    """Build the static [ConsistentInitPlan][] for a solver problem."""
    return ConsistentInitPlan(
        problem,
        plan,
        tol=tol,
        max_iters=max_iters,
        chord_iters=chord_iters,
        backend=backend,
        analysis=analysis,
        route=route,
        layout=layout,
        precision=precision,
        reverse_precision=reverse_precision,
    )


def _embed(icp: ConsistentInitPlan, y_pinned, z):
    """Blend the state-aligned joint unknowns into the full ``(y, ydot)`` pair.

    Slot ``s`` of ``z`` is ``y[s]`` for an algebraic variable and ``ydot[s]``
    for a differential one; the algebraic slopes are structurally zero.
    """
    mask = jnp.asarray(icp.diff_mask)
    y = jnp.where(mask, y_pinned, z)
    ydot = jnp.where(mask, z, jnp.zeros_like(z))
    return y, ydot


def _k_matrix(icp: ConsistentInitPlan, t, y, ydot, theta, control):
    """Exact dense ``K`` at a point (state-aligned columns).

    Column ``s`` is ``F_y[:, s]`` for an algebraic variable and
    ``F_ydot[:, s]`` for a differential one. The dense-route Jacobian and the
    validation/diagnostic reference; the sparse route uses
    [_k_values][diff_soup.problem.implicit.consistent_init._k_values]
    and never calls this.
    """
    plan = icp.plan
    fy = plan.fy_values(t, y, ydot, theta, control)
    fydot = plan.fydot_values(t, y, ydot, theta, control)
    if plan.mode == "dense":
        return jnp.where(jnp.asarray(icp.diff_mask)[None, :], fydot, fy)
    k = jnp.zeros((icp.n, icp.n), dtype=jnp.result_type(fy, fydot))
    k = k.at[icp._fy_rows, icp._fy_cols].add(fy[icp._fy_keep], unique_indices=True)
    return k.at[icp._fyd_rows, icp._fyd_cols].add(fydot, unique_indices=True)


def _k_values(icp: ConsistentInitPlan, t, y, ydot, theta, control):
    """Exact ``K`` values for the backend route, never an ``(n, n)`` array.

    Sparse-mode plans return values over
    [ConsistentInitPlan.k_topology][diff_soup.problem.implicit.consistent_init.ConsistentInitPlan.k_topology]
    (the kept ``F_y`` algebraic-column entries followed by the ``F_ydot``
    entries); dense-mode plans return the dense ``K`` flattened row-major
    over the full pattern.
    """
    plan = icp.plan
    if plan.mode == "dense":
        return _k_matrix(icp, t, y, ydot, theta, control).reshape(-1)
    fy = jnp.asarray(plan.fy_values(t, y, ydot, theta, control))
    fydot = jnp.asarray(plan.fydot_values(t, y, ydot, theta, control))
    dtype = jnp.result_type(fy, fydot)
    return jnp.concatenate([fy[icp._fy_keep].astype(dtype), fydot.astype(dtype)])


@functools.lru_cache(maxsize=64)
def _full_topology(n: int) -> SparseTopology:
    """The dense ``n x n`` pattern the dense route factors on."""
    return SparseTopology.full(n)


def _routes(icp: ConsistentInitPlan, values, n: int):
    """The forward and reverse routes of matrix values on the plan's linear route.

    The sparse route serves values over the plan's pattern with its backend
    (the reverse one analysed with the plan's reverse policy, at its adjoint
    dtype); the dense route serves a dense matrix (flattened row-major) with
    the dense backend.
    """
    values = jnp.asarray(values)
    reverse_dtype = values.dtype if icp.adjoint_dtype is None else jnp.dtype(icp.adjoint_dtype)
    if icp.linear_backend is None:
        topology = _full_topology(n)
        return (
            linear_route(None, None, dtype=values.dtype, topology=topology),
            linear_route(None, None, dtype=reverse_dtype, topology=topology),
        )
    return (
        linear_route(icp.linear_backend, icp.linear_analysis, dtype=values.dtype),
        linear_route(icp.linear_backend, icp.reverse_linear_analysis, dtype=reverse_dtype),
    )


def _route_factor(icp: ConsistentInitPlan, values, n: int) -> RouteFactor:
    """Factor matrix values once on the plan's forward linear route."""
    route, _reverse = _routes(icp, values, n)
    return RouteFactor(route, jnp.asarray(values))


def _root_tangent(icp: ConsistentInitPlan, values, n: int, rhs):
    """The tangent solve ``x = M^-1 rhs`` at the root.

    A [transposable_solve][diff_soup.nonlinear._kkt.transposable_solve]
    (linear in ``rhs``; JAX's transposition of it factors ``M^T`` directly on
    the plan's reverse route at its adjoint dtype).
    """
    values = jnp.asarray(values)
    route, reverse = _routes(icp, values, n)
    return transposable_solve(
        route, values, rhs, reverse_route=reverse, adjoint_dtype=icp.adjoint_dtype
    )


def _root_solvable(icp: ConsistentInitPlan, values, n: int):
    """Whether the matrix at the root solves.

    The checked solve of a constant right-hand side with a factor of ``M``
    (see [RouteFactor.solves][diff_soup.nonlinear._kkt.RouteFactor.solves]), a
    function of primal values only, so a rule that gates on it stays linear
    in the tangents.
    """
    route, _reverse = _routes(icp, jnp.asarray(values), n)
    return RouteFactor(route, jnp.asarray(values)).solves()


def _zero_where_not(keep, tree):
    """Replace every inexact leaf of ``tree`` with exact zeros where ``keep`` is false.

    A selection, not a product: an infinite or NaN leaf that is not kept
    becomes zero instead of NaN, and so does its cotangent under
    transposition. Integer (``float0``) leaves and ``None`` pass unchanged.
    """
    keep = jnp.asarray(keep)

    def gate(leaf):
        if getattr(leaf, "dtype", None) == jax.dtypes.float0:
            return leaf
        return jnp.where(keep, leaf, jnp.zeros_like(leaf))

    return jax.tree_util.tree_map(gate, tree)


def _gated_root_tangent(icp: ConsistentInitPlan, values, n: int, rhs_dot, valid):
    """The tangent solve ``M dz = -rhs_dot`` with its right-hand side and result gated on ``valid``.

    On an invalid root both are exact zeros, so the transposed solve neither
    receives nor returns a cotangent there: a singular matrix cannot turn a
    discarded cotangent into NaN.
    """
    rhs = _zero_where_not(valid, -rhs_dot)
    dz = _root_tangent(icp, values, n, rhs)
    return _zero_where_not(valid, dz)


def _k_root_values(icp: ConsistentInitPlan, t, y, ydot, theta, control):
    """Exact ``K`` values in the layout of the plan's linear route."""
    if icp.linear_backend is None:
        return _k_matrix(icp, t, y, ydot, theta, control).reshape(-1)
    return _k_values(icp, t, y, ydot, theta, control)


def _k_solver(icp, t, y, ydot, theta, control):
    """Factorize the exact ``K`` once at a point; return its solve.

    Routes through the plan's resolved linear backend when one is recorded
    (sparse values over the joint topology) and through the dense backend
    otherwise.
    """
    held = _route_factor(icp, _k_root_values(icp, t, y, ydot, theta, control), icp.n)
    return held.solve


#: Chord contraction gate: a chord step whose residual norm exceeds this
#: fraction of the previous norm hands the solve to fresh-Jacobian Newton.
_CHORD_CONTRACTION_LIMIT = 0.5


def _phased_newton(icp: ConsistentInitPlan, z0, residual_at, make_solver):
    """Chord phase over one frozen factor, then fresh-Jacobian exact Newton.

    The chord phase factors the iteration matrix once at the seed and reuses
    that factor for up to ``chord_iters`` updates, watching per-iteration
    contraction and finiteness; poor contraction, a nonfinite update or
    residual, or an exhausted chord budget hands the best iterate so far to
    an exact-Newton phase that refreshes the matrix every iteration within
    the remaining ``max_iters`` budget. Convergence is always judged on a
    freshly evaluated residual, and an unconverged solve reports itself
    honestly. A seed that already satisfies the residual gate (or is
    nonfinite) returns without factoring anything.
    """
    r0 = residual_at(z0)
    norm0 = jnp.linalg.norm(r0)
    it0 = jnp.asarray(0, jnp.int32)
    chord_budget = min(icp.chord_iters, icp.max_iters)

    def solve_phases(_):
        """Run the chord phase, then exact Newton on whatever remains."""
        if chord_budget > 0:
            # One factorization before the chord loop serves every chord update.
            solve0 = make_solver(z0)

            def chord_going(carry):
                """Iterate while unconverged, in budget, finite, untriggered."""
                _z, r, it, stop = carry
                norm = jnp.linalg.norm(r)
                return (~stop) & (it < chord_budget) & (norm > icp.tol) & jnp.isfinite(norm)

            def chord_step(carry):
                """One frozen-factor update with contraction/finiteness guards."""
                z, r, it, _stop = carry
                z_new = z - solve0(r)
                r_new = residual_at(z_new)
                norm_old = jnp.linalg.norm(r)
                norm_new = jnp.linalg.norm(r_new)
                finite = jnp.isfinite(norm_new) & jnp.all(jnp.isfinite(z_new))
                improved = finite & (norm_new < norm_old)
                trigger = (~finite) | (norm_new > _CHORD_CONTRACTION_LIMIT * norm_old)
                z_keep = jnp.where(improved, z_new, z)
                r_keep = jnp.where(improved, r_new, r)
                return z_keep, r_keep, it + 1, trigger

            z, r, it, _ = jax.lax.while_loop(
                chord_going, chord_step, (z0, r0, it0, jnp.asarray(False))
            )
        else:
            z, r, it = z0, r0, it0

        def exact_going(carry):
            """Iterate while unconverged, within budget, and finite."""
            _z, r, it = carry
            norm = jnp.linalg.norm(r)
            return (it < icp.max_iters) & (norm > icp.tol) & jnp.isfinite(norm)

        def exact_step(carry):
            """One exact Newton update with a fresh iteration matrix."""
            z, r, it = carry
            solve = make_solver(z)
            z_new = z - solve(r)
            return z_new, residual_at(z_new), it + 1

        return jax.lax.while_loop(exact_going, exact_step, (z, r, it))

    def already_done(_):
        """The seed already meets the gate (or is nonfinite): no solve."""
        return z0, r0, it0

    z, r, iterations = jax.lax.cond(
        (norm0 > icp.tol) & jnp.isfinite(norm0), solve_phases, already_done, None
    )
    norm = jnp.linalg.norm(r)
    converged = (norm <= icp.tol) & jnp.isfinite(norm)
    return z, norm, iterations, converged


def _newton_primal(icp: ConsistentInitPlan, t0, y0, ydot_guess, theta, control):
    """Bounded chord-then-exact Newton solve for the joint unknowns.

    Never differentiated; the derivative entries apply the implicit-function
    rule at the converged root.
    """
    if icp.route != "joint":
        raise ValueError(
            "this entry runs the joint initialization solve but the plan was "
            "built for the reduced route; build the plan with route='joint', "
            "or solve through consistent_initial_state, which dispatches on "
            "the plan's own route."
        )
    z0 = jnp.where(jnp.asarray(icp.diff_mask), ydot_guess, y0)

    def residual_at(z):
        """The full residual at the embedded joint iterate."""
        y, ydot = _embed(icp, y0, z)
        return icp.plan._residual(t0, y, ydot, theta, control)

    def make_solver(z):
        """Factor the exact joint matrix at an iterate; return the solve."""
        y, ydot = _embed(icp, y0, z)
        return _k_solver(icp, t0, y, ydot, theta, control)

    z, norm, iterations, converged = _phased_newton(icp, z0, residual_at, make_solver)
    y_star, ydot_star = _embed(icp, y0, z)
    return y_star, ydot_star, norm, iterations, converged


def _pinned_residual_fn(icp: ConsistentInitPlan, t0, y_star, ydot_star):
    """The residual at the frozen root as a function of the differentiable inputs."""
    diff_mask = jnp.asarray(icp.diff_mask)

    def pinned_residual(y0_in, theta_in, control_in):
        """Residual with pinned rows from ``y0_in`` and solved values frozen."""
        y = jnp.where(diff_mask, y0_in, y_star)
        return icp.plan._residual(t0, y, ydot_star, theta_in, control_in)

    return pinned_residual


def _zero_cotangent(x):
    """Zero cotangent (or tangent) matching ``x``'s pytree and leaf avals."""
    return jax.tree_util.tree_map(lambda leaf: zero_from_primal(leaf, symbolic_zeros=False), x)


def _poison(x, valid):
    """Replace ``x`` with NaN when the root or its sensitivity solve is invalid."""
    return jnp.where(valid, x, jnp.full_like(x, jnp.nan))


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def consistent_ic_root(icp, t0, y0, ydot_guess, theta, control):
    """Joint consistent-initialization root with its implicit-function derivative.

    Returns ``(y, ydot, residual_norm, iterations, converged)``. Tangents come
    from the implicit-function rule at the converged root; reverse mode is
    JAX's transposition of that rule, whose cotangent solve factors ``K^T``
    directly on the plan's reverse policy. Diagnostics are gradient-dead.
    """
    return _newton_primal(icp, t0, y0, ydot_guess, theta, control)


@consistent_ic_root.defjvp
def _consistent_ic_root_jvp(icp, primals, tangents):
    """Tangent: exact implicit solve in the joint unknowns at the root."""
    t0, y0, ydot_guess, theta, control = primals
    _dt0, dy0, _dydot_guess, dtheta, dcontrol = tangents
    # The primal through the entry itself: differentiating this rule again
    # differentiates the root by this rule, never through the Newton loop.
    out = consistent_ic_root(icp, t0, y0, ydot_guess, theta, control)
    y_star, ydot_star, norm, iterations, converged = out
    diff_mask = jnp.asarray(icp.diff_mask)

    values = _k_root_values(icp, t0, y_star, ydot_star, theta, control)
    valid = jnp.logical_and(converged, _root_solvable(icp, values, icp.n))
    pinned_residual = _pinned_residual_fn(icp, t0, y_star, ydot_star)
    dy0_pinned = _zero_where_not(valid, jnp.where(diff_mask, dy0, 0.0))
    dtheta, dcontrol = _zero_where_not(valid, (dtheta, dcontrol))
    _, rhs_dot = jax.jvp(pinned_residual, (y0, theta, control), (dy0_pinned, dtheta, dcontrol))
    dz = _gated_root_tangent(icp, values, icp.n, rhs_dot, valid)

    dy = jnp.where(diff_mask, dy0, dz)
    dydot = jnp.where(diff_mask, dz, jnp.zeros_like(dz))
    return out, (
        _poison(dy, valid),
        _poison(dydot, valid),
        _zero_cotangent(norm),
        _zero_cotangent(iterations),
        _zero_cotangent(converged),
    )


def _g_matrix(icp: ConsistentInitPlan, t0, y, theta, control):
    """Exact dense algebraic block ``G = F_y[a, a]`` at a point.

    The dense-route Jacobian of the reduced algebraic solve; the sparse
    route uses
    [_g_values][diff_soup.problem.implicit.consistent_init._g_values]
    and never materializes this.
    """
    plan = icp.plan
    zeros = jnp.zeros_like(y)
    fy = plan.fy_values(t0, y, zeros, theta, control)
    if plan.mode == "dense":
        a = jnp.asarray(icp.a_idx)
        return fy[a][:, a]
    g = jnp.zeros((icp.n_a, icp.n_a), dtype=jnp.asarray(fy).dtype)
    return g.at[icp._ra_rows, icp._ra_cols].add(jnp.asarray(fy)[icp._ra_keep], unique_indices=True)


def _g_values(icp: ConsistentInitPlan, t0, y, theta, control):
    """Exact algebraic-block values for the backend route.

    Sparse-mode plans return values over
    [ConsistentInitPlan.reduced_topology][diff_soup.problem.implicit.consistent_init.ConsistentInitPlan.reduced_topology]; dense-mode plans return the
    dense block flattened row-major over the full pattern.
    """
    plan = icp.plan
    if plan.mode == "dense":
        return _g_matrix(icp, t0, y, theta, control).reshape(-1)
    zeros = jnp.zeros_like(y)
    fy = jnp.asarray(plan.fy_values(t0, y, zeros, theta, control))
    return fy[icp._ra_keep]


def _g_root_values(icp: ConsistentInitPlan, t0, y, theta, control):
    """Exact algebraic-block values in the layout of the plan's linear route."""
    if icp.linear_backend is None:
        return _g_matrix(icp, t0, y, theta, control).reshape(-1)
    return _g_values(icp, t0, y, theta, control)


def _g_solver(icp, t0, y, theta, control):
    """Factorize the exact algebraic block once at a point; return its solve.

    Routes through the plan's resolved linear backend when one is recorded
    and through the dense backend otherwise, mirroring
    [_k_solver][diff_soup.problem.implicit.consistent_init._k_solver] on the
    compacted algebraic block.
    """
    held = _route_factor(icp, _g_root_values(icp, t0, y, theta, control), icp.n_a)
    return held.solve


def _reduced_newton_primal(icp: ConsistentInitPlan, t0, y0, theta, control):
    """Bounded chord-then-exact Newton solve on the algebraic rows.

    Never differentiated. The unknowns are the local- and global-algebraic values;
    the residual is the algebraic restriction of the full
    residual at zero slopes, and the iteration matrix is the exact algebraic
    block ``F_y[a, a]``.
    """
    if icp.route != "reduced":
        raise ValueError(
            "this entry runs the reduced initialization solve but the plan "
            "was built for the joint route; build the plan with "
            "route='reduced' (or 'auto' on an eligible problem), or solve "
            "through consistent_initial_state, which dispatches on the "
            "plan's own route."
        )
    y0 = jnp.asarray(y0)
    a_idx = jnp.asarray(icp.a_idx)
    zeros = jnp.zeros_like(y0)
    z0 = y0[a_idx]

    def residual_at(z):
        """The algebraic residual rows at the embedded iterate."""
        y = y0.at[a_idx].set(z, unique_indices=True, indices_are_sorted=True)
        return icp.plan._residual(t0, y, zeros, theta, control)[a_idx]

    def make_solver(z):
        """Factor the exact algebraic block at an iterate; return the solve."""
        y = y0.at[a_idx].set(z, unique_indices=True, indices_are_sorted=True)
        return _g_solver(icp, t0, y, theta, control)

    z, _norm, iterations, converged = _phased_newton(icp, z0, residual_at, make_solver)
    y_star = y0.at[a_idx].set(z, unique_indices=True, indices_are_sorted=True)
    return y_star, iterations, converged


def _pinned_alg_residual_fn(icp: ConsistentInitPlan, t0, y_star):
    """The frozen-root algebraic rows as a function of the differentiable inputs."""
    diff_mask = jnp.asarray(icp.diff_mask)
    a_idx = jnp.asarray(icp.a_idx)
    zeros = jnp.zeros_like(y_star)

    def pinned_residual(y0_in, theta_in, control_in):
        """Algebraic rows with pinned values from ``y0_in``, solved values frozen."""
        y = jnp.where(diff_mask, y0_in, y_star)
        return icp.plan._residual(t0, y, zeros, theta_in, control_in)[a_idx]

    return pinned_residual


@functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
def _reduced_ic_root(icp, t0, y0, theta, control):
    """Reduced algebraic root with its implicit-function derivative.

    Returns ``(y, iterations, converged)`` where ``y`` carries the pinned
    differential values and the solved algebraic values. Tangents come from
    the implicit-function rule on the algebraic block at the converged root
    (reverse mode is its transposition); diagnostics are gradient-dead.
    """
    return _reduced_newton_primal(icp, t0, y0, theta, control)


@_reduced_ic_root.defjvp
def _reduced_ic_root_jvp(icp, primals, tangents):
    """Tangent: exact implicit solve on the algebraic block at the root."""
    t0, y0, theta, control = primals
    _dt0, dy0, dtheta, dcontrol = tangents
    # The primal through the entry itself, as in the joint rule.
    out = _reduced_ic_root(icp, t0, y0, theta, control)
    y_star, iterations, converged = out
    diff_mask = jnp.asarray(icp.diff_mask)
    a_idx = jnp.asarray(icp.a_idx)

    values = _g_root_values(icp, t0, y_star, theta, control)
    valid = jnp.logical_and(converged, _root_solvable(icp, values, icp.n_a))
    pinned_residual = _pinned_alg_residual_fn(icp, t0, y_star)
    dy0_pinned = _zero_where_not(valid, jnp.where(diff_mask, dy0, 0.0))
    dtheta, dcontrol = _zero_where_not(valid, (dtheta, dcontrol))
    _, rhs_dot = jax.jvp(pinned_residual, (y0, theta, control), (dy0_pinned, dtheta, dcontrol))
    dz = _gated_root_tangent(icp, values, icp.n_a, rhs_dot, valid)

    dz_full = jnp.zeros_like(y_star).at[a_idx].set(dz, unique_indices=True, indices_are_sorted=True)
    dy = jnp.where(diff_mask, dy0, dz_full)
    return out, (
        _poison(dy, valid),
        _zero_cotangent(iterations),
        _zero_cotangent(converged),
    )


def _reconstruct_slopes(icp: ConsistentInitPlan, t0, y, theta, control):
    """Reconstruct the differential slopes ``ydot_d = f_d / D_dd``.

    Plain differentiable code: the diagonal comes from the derivative
    operator applied to ones (exact for diagonal support) and ``f`` from the
    residual at zero slopes. Invertibility is a value property checked here,
    at the solved point: a non-finite or zero diagonal entry on a
    differential slot fails loudly — the returned slopes are NaN and
    ``diag_ok`` is false. ``t0`` carries no derivative.
    """
    t0 = jax.lax.stop_gradient(t0)
    diff_mask = jnp.asarray(icp.diff_mask)
    diag = icp._derivative_apply(t0, y, theta, control, jnp.ones_like(y))
    f = -icp.plan._residual(t0, y, jnp.zeros_like(y), theta, control)
    on_diff = jnp.isfinite(diag) & (diag != 0.0)
    diag_ok = jnp.all(jnp.where(diff_mask, on_diff, True))
    safe_diag = jnp.where(diff_mask & on_diff, diag, jnp.ones_like(diag))
    ydot = jnp.where(diff_mask, f / safe_diag, 0.0)
    return _poison(ydot, diag_ok), diag_ok


def _reduced_root_and_slopes(icp: ConsistentInitPlan, root, t0, y0, ydot_guess, theta, control):
    """Compose one reduced root entry with slope reconstruction.

    Same five outputs as the joint entries. The slope guess is unused: the
    reduced route reconstructs the slopes exactly, so the guess carries no
    derivative by construction. The reported norm is the full residual at
    the composed point (the common post-solve metric), gradient-dead.
    """
    del ydot_guess
    if icp.n_a == 0:
        y_star = jnp.asarray(y0)
        iterations = jnp.asarray(0, jnp.int32)
        alg_ok = jnp.asarray(True)
    else:
        y_star, iterations, alg_ok = root(icp, t0, y0, theta, control)
    ydot, diag_ok = _reconstruct_slopes(icp, t0, y_star, theta, control)
    r = icp.plan._residual(t0, y_star, ydot, theta, control)
    norm = jax.lax.stop_gradient(jnp.linalg.norm(r))
    converged = alg_ok & diag_ok & jnp.isfinite(norm)
    return y_star, ydot, norm, iterations, converged


def consistent_ic_root_reduced(icp, t0, y0, ydot_guess, theta, control):
    """Reduced consistent-initialization solve with its implicit-function derivative.

    Available for a diagonal invertible mass only (see
    [reduced_route_eligible][diff_soup.problem.implicit.consistent_init.reduced_route_eligible]): the algebraic values solve on the exact
    algebraic block and the differential slopes reconstruct from the mass
    diagonal. Same outputs and derivative contract as
    [consistent_ic_root][diff_soup.problem.implicit.consistent_init.consistent_ic_root].
    """
    return _reduced_root_and_slopes(icp, _reduced_ic_root, t0, y0, ydot_guess, theta, control)


def _solver_problem_and_theta(problem, theta):
    """Normalize an authored source/view and resolve its default parameters."""
    from diff_soup.problem.implicit.problems import ImplicitProblem, MassProblem

    if isinstance(problem, (MassProblem, ImplicitProblem)):
        default_theta = getattr(problem, "default_theta", None)
        return problem, default_theta() if theta is None and callable(default_theta) else theta
    from diff_soup.problem.implicit.solver_problem import get_or_build_solver_view

    view = get_or_build_solver_view(problem)
    return view.problem, view.default_theta() if theta is None else theta


def consistent_initial_state(
    problem,
    t0,
    y0,
    theta,
    control=None,
    *,
    ydot0=None,
    plan=None,
    init_plan: Optional[ConsistentInitPlan] = None,
    tol: float = 1e-12,
    max_iters: int = 30,
    chord_iters: int = 8,
    backend="auto",
    analysis=None,
    route: str = "auto",
) -> ConsistentInitResult:
    """Solve for a consistent initial state of a DAE at `t0`.

    It holds the differential values of `y0` and solves for the algebraic values and the
    differential slopes with Newton's method. The result is differentiable with respect to
    `y0` and `theta`.

    Args:
        problem: The problem from `spec.build()`, or a `MassProblem` or `ImplicitProblem`.
        t0: The initial time.
        y0: The initial state, whose algebraic entries are the starting guess.
        theta: The parameters, or `None` for the problem's defaults.
        control: The control values, or `None`.
        ydot0: A guess for the slopes, zeros by default.
        plan: A prebuilt linearization plan, or `None` to build one.
        init_plan: A prebuilt `ConsistentInitPlan`, which overrides every setting below it.
        tol: The residual 2-norm at which Newton's method stops.
        max_iters: The total Newton iteration budget.
        chord_iters: The first iterations, which reuse one factorization before each later
            iteration refactors.
        backend: The linear backend, a name as on `linear=` or a `ds.linear` object.
        analysis: A prebuilt linear analysis of the Newton matrix, or `None`.
        route: `"reduced"` solves the algebraic block alone and needs a diagonal mass, `"joint"`
            the full system, and `"auto"` picks.

    Returns:
        A `ConsistentInitResult`.
    """
    problem, theta = _solver_problem_and_theta(problem, theta)
    icp = (
        init_plan
        if init_plan is not None
        else ConsistentInitPlan(
            problem,
            plan,
            tol=tol,
            max_iters=max_iters,
            chord_iters=chord_iters,
            backend=backend,
            analysis=analysis,
            route=route,
        )
    )
    y0 = jnp.asarray(y0)
    ydot_guess = jnp.zeros_like(y0) if ydot0 is None else jnp.asarray(ydot0)
    out = _plan_root_entry(icp)(t0, y0, ydot_guess, theta, control)
    y, ydot, norm, iterations, converged = out
    return ConsistentInitResult(
        y=y, ydot=ydot, residual_norm=norm, iterations=iterations, converged=converged
    )


def _partition_note(structure, index: int) -> str:
    """A ``" (partition 'name')"`` annotation for a state index, or ``""``."""
    names = [name for name, lo, hi in structure.partitions if lo <= index < hi]
    if not names:
        return ""
    return " (partition " + ", ".join(repr(name) for name in names) + ")"


def _describe_unknowns(icp: ConsistentInitPlan, z_indices) -> str:
    """Name joint unknowns: algebraic values ``y[i]`` and slopes ``ydot[i]``."""
    values = [int(j) for j in z_indices if not icp.diff_mask[int(j)]]
    slopes = [int(j) for j in z_indices if icp.diff_mask[int(j)]]
    parts = []
    if values:
        parts.append(
            "algebraic values "
            + ", ".join(f"y[{i}]{_partition_note(icp.structure, i)}" for i in values)
        )
    if slopes:
        parts.append(
            "differential slopes "
            + ", ".join(f"ydot[{i}]{_partition_note(icp.structure, i)}" for i in slopes)
        )
    return "; ".join(parts) if parts else "none identified"


def check_consistent_init(
    problem,
    t0,
    y0,
    theta,
    control=None,
    *,
    ydot0=None,
    plan=None,
    init_plan: Optional[ConsistentInitPlan] = None,
    tol: float = 1e-12,
    max_iters: int = 30,
) -> ConsistentInitResult:
    """Solve for a consistent initial state, and raise with a diagnosis when there is none.

    It runs the joint route with a dense linear solve and then checks the rank of the joint
    matrix at the result. It runs on concrete arrays, outside `jax.jit`.

    Args:
        problem: The problem from `spec.build()`, or a `MassProblem` or `ImplicitProblem`.
        t0: The initial time.
        y0: The initial state, whose algebraic entries are the starting guess.
        theta: The parameters, or `None` for the problem's defaults.
        control: The control values, or `None`.
        ydot0: A guess for the slopes, zeros by default.
        plan: A prebuilt linearization plan, or `None` to build one.
        init_plan: A prebuilt `ConsistentInitPlan`, which overrides every setting below it.
        tol: The residual 2-norm at which Newton's method stops.
        max_iters: The total Newton iteration budget.

    Returns:
        A converged `ConsistentInitResult`.

    Raises:
        ValueError: When the joint matrix is singular, naming the unknowns and rows involved, or
            when Newton's method does not converge.
    """
    problem, theta = _solver_problem_and_theta(problem, theta)
    icp = (
        init_plan
        if init_plan is not None
        else ConsistentInitPlan(
            problem, plan, tol=tol, max_iters=max_iters, backend="dense", route="joint"
        )
    )
    if icp.route != "joint":
        # The diagnostic always analyzes the joint matrix (it contains the
        # reduced solve's algebraic block); re-express the plan as its joint
        # equivalent.
        icp = ConsistentInitPlan(
            problem,
            icp.plan,
            tol=icp.tol,
            max_iters=icp.max_iters,
            backend="dense",
            route="joint",
        )
    y0 = jnp.asarray(y0)
    ydot_guess = jnp.zeros_like(y0) if ydot0 is None else jnp.asarray(ydot0)
    y_star, ydot_star, norm, iterations, converged = _newton_primal(
        icp, t0, y0, ydot_guess, theta, control
    )
    k = np.asarray(_k_matrix(icp, t0, y_star, ydot_star, theta, control))
    n = icp.n
    if not np.all(np.isfinite(k)):
        raise ValueError(
            "consistent initialization cannot be analyzed: the joint matrix "
            "(the algebraic columns of F_y and the differential columns of "
            "F_ydot) has non-finite entries at the solved point: the "
            "residual's derivatives are not finite there. Supply seeds inside "
            "the residual's domain."
        )
    u, s, vh = np.linalg.svd(k)
    cutoff = (s[0] if s.size else 0.0) * n * np.finfo(k.dtype).eps
    rank = int(np.sum(s > cutoff))
    if rank < n:
        null_cols = vh[rank:]
        col_scale = np.max(np.abs(null_cols), axis=0)
        implicated_z = np.flatnonzero(col_scale > 1e-6 * max(col_scale.max(), 1e-300))
        row_scale = np.max(np.abs(u[:, rank:]), axis=1)
        implicated_rows = np.flatnonzero(row_scale > 1e-6 * max(row_scale.max(), 1e-300))
        rows_text = ", ".join(
            f"{int(i)}{_partition_note(icp.structure, int(i))}" for i in implicated_rows
        )
        raise ValueError(
            "consistent initialization is singular at the solved point: the "
            "joint matrix (the algebraic columns of F_y and the differential "
            f"columns of F_ydot) has rank {rank} of {n}. Implicated unknowns: "
            f"{_describe_unknowns(icp, implicated_z)}. Implicated equation "
            f"rows: [{rows_text}]. Every algebraic value and differential "
            "slope must be uniquely determined by the residual at t0: remove "
            "redundant equations, correct the declared variable roles, or add "
            "the missing coupling."
        )
    if not bool(converged):
        remedy = (
            "Raise max_iters, loosen tol, or supply closer algebraic-value and slope seeds."
            if jnp.asarray(y_star).dtype != jnp.float32
            else "In float32 the residual's own rounding can exceed tol; loosen tol, or "
            "initialize in float64 with Precision.fp32_ir() or Precision.fp64()."
        )
        raise ValueError(
            "consistent initialization did not converge: after "
            f"{int(iterations)} Newton iterations the residual 2-norm is "
            f"{float(norm):.3e} against the tolerance {icp.tol:.1e}, with a "
            f"nonsingular joint matrix at the last iterate. {remedy}"
        )
    return ConsistentInitResult(
        y=y_star,
        ydot=ydot_star,
        residual_norm=norm,
        iterations=iterations,
        converged=converged,
    )
