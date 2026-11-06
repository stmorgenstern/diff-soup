"""The plan of one initialization solve: its route, its linear backend and its index maps.

A [ConsistentInitPlan][diff_soup.problem.implicit.consistent_init.ConsistentInitPlan]
holds the static machinery of one problem's initialization: the route (the joint
state-aligned system, or the algebraic-block elimination a diagonal mass allows), the
linear backend chosen for the route's system size and platform, and the index maps that
assemble the exact Jacobian and the relative gate's scale from the linearization plan.
"""

from __future__ import annotations

import numpy as np

from diff_soup.linear import LinearSolver, SparseTopology
from diff_soup.linear._resolver import BackendResolver, resolve_backend
from diff_soup.nonlinear._kkt import LinearPlan
from diff_soup.problem.implicit.linearize import build_linearization_plan
from diff_soup.utils.placement import default_platform


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


def initialization_route(problem, route: str = "auto") -> tuple[str, int]:
    """Resolve the initialization solve's route and the size of its linear system.

    Args:
        problem: The solver problem.
        route: ``"auto"`` (the reduced route exactly when the problem is
            eligible), ``"joint"`` or ``"reduced"``.

    Returns:
        The route, ``"joint"`` or ``"reduced"``, and its system size: the
        state count on the joint route, the algebraic count on the reduced one.

    Raises:
        ValueError: If ``route`` is not one of the three names, or
            ``"reduced"`` is requested for a problem without a diagonal mass.
    """
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
    structure = problem.structure
    block = int(structure.algebraic_indices.size) if route == "reduced" else int(structure.n)
    return route, block


def resolve_backend_route(backend, *, plan_mode: str, n_dofs: int):
    """Resolve a backend request to its concrete route ``(name, instance)``.

    ``backend`` is ``"auto"``, a concrete name (``"dense"`` / ``"klu"`` /
    ``"superlu"`` / ``"cudss"``), or a
    [LinearSolver][diff_soup.linear.LinearSolver] instance.
    ``"auto"`` stays dense for a dense-mode linearization plan (the joint
    matrix is assembled densely regardless) and otherwise picks the
    platform's route from the system size. A concrete name resolves to the
    per-process instance the engines share
    ([resolve_backend][diff_soup.linear._resolver.resolve_backend]); the
    returned ``instance`` is ``None`` on the dense route. Two requests that resolve to the same name
    are interchangeable for the joint initialization solve: the backend
    instances hold no per-solve state, and their symbolic analyses are
    memoised process-globally on topology content. An iterative backend
    ([Krylov][diff_soup.linear.Krylov]) resolves to its exact inner backend:
    the initialization's Newton steps and its implicit-function derivative
    solve with exact inverses. A backend whose bound ``layout.n_dofs`` differs
    from the initialization system size resolves as ``"auto"`` does, by size
    and platform; the time-stepping backend is unchanged.
    """
    if isinstance(backend, LinearSolver):
        from diff_soup.linear._krylov import exact_backend

        backend = exact_backend(backend)
        bound_size = getattr(getattr(backend, "layout", None), "n_dofs", None)
        if bound_size is None or int(bound_size) == n_dofs:
            name = backend.backend_name
            return name, (None if name == "dense" else backend)
        backend = "auto"
    if isinstance(backend, str):
        if backend == "auto" and plan_mode == "dense":
            return "dense", None
        platform = default_platform()
        name = BackendResolver(backend=backend).choose(block_size=n_dofs, platform=platform)
        return name, (None if name == "dense" else resolve_backend(name))
    raise ValueError(
        f"backend must be 'auto', 'dense', 'klu', 'superlu', 'cudss', or a "
        f"LinearSolver instance; got {backend!r}."
    )


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
        rtol: float = 0.0,
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
            tol: Absolute residual 2-norm target for Newton iterations.
            rtol: Relative part of the final residual gate. A positive value
                also accepts a residual 2-norm up to ``tol + (rtol + γ) S``,
                with ``S = ‖|F_y| |y| + |F_ydot| |ydot|‖₂`` at the final
                iterate and ``γ`` the rounding bound of the longest residual
                row. Zero keeps the absolute gate only. Newton aims at
                ``tol`` either way.
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
        self.rtol = float(rtol)
        if not np.isfinite(self.rtol) or self.rtol < 0.0:
            raise ValueError("rtol must be finite and nonnegative")
        self.max_iters = int(max_iters)
        self.chord_iters = int(chord_iters)
        self.layout = layout
        self.precision = precision
        self.reverse_precision = reverse_precision
        policy = precision if reverse_precision is None else reverse_precision
        self.adjoint_dtype = getattr(policy, "adjoint_compute_dtype", None)
        self.structure = structure

        route, self.block = initialization_route(problem, route)
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

        self._scale_terms, self._scale_length = _scale_terms(self)
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
        block = self.block
        if block == 0:
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


def build_consistent_init_plan(
    problem,
    plan=None,
    *,
    tol: float = 1e-12,
    rtol: float = 0.0,
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
        rtol=rtol,
        max_iters=max_iters,
        chord_iters=chord_iters,
        backend=backend,
        analysis=analysis,
        route=route,
        layout=layout,
        precision=precision,
        reverse_precision=reverse_precision,
    )


def _scale_terms(icp: ConsistentInitPlan):
    """The terms of the residual scale on the rows the solve drives to zero.

    Returns, for ``F_y`` and (joint route only) ``F_ydot``, the indices of the
    pattern entries in those rows and their row numbers among them, and the
    longest such row. The reduced route keeps the algebraic rows and leaves
    out ``F_ydot``, whose slopes are zero there.
    """
    plan = icp.plan
    n_rows = icp.n_a if icp.route == "reduced" else icp.n
    row_of = np.full(icp.n, -1, dtype=np.int64)
    row_of[np.asarray(icp.a_idx) if icp.route == "reduced" else np.arange(icp.n)] = np.arange(
        n_rows
    )
    blocks = (("fy", plan.fy_topology),)
    if icp.route == "joint":
        blocks += (("fydot", plan.fydot_topology),)
    terms, length = [], np.zeros(n_rows, dtype=np.int64)
    for name, topology in blocks:
        rows = row_of[np.asarray(topology.rows)] if plan.mode == "sparse" else None
        if rows is None:
            length += icp.n
            terms.append((name, None, None))
            continue
        keep = np.flatnonzero(rows >= 0)
        np.add.at(length, rows[keep], 1)
        terms.append((name, keep, rows[keep]))
    return terms, int(length.max(initial=0))
