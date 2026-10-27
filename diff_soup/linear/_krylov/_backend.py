"""The Krylov backend: GMRES on the current operator, preconditioned by a held exact factor.

A factor of [Krylov][diff_soup.linear.Krylov] is a preconditioner of a
current operator. Its ``values`` are always the operator's (the matrix the
solves apply); its handle holds a factor of an exact inner backend (cuDSS,
SuperLU_MT, KLU, Dense), which is rebuilt only where a refactor's predicate
asks. A solve runs restarted GMRES (``diff_soup.linear._krylov._gmres``) on
the operator of the factor's values, right-preconditioned by a solve with the
held inner factor, and stops at a [Tolerance][diff_soup.linear.Tolerance].

The inner factor of a factor of orientation ``"A"`` factors ``A_bc``; that of
a factor of orientation ``"AT"`` is a primary factor of ``A_bc^T``, analysed
on the transposed pattern
([transposed_topology][diff_soup.linear._krylov.transposed_topology]), so the
solves of an ``"AT"`` factor precondition with solves that are not transposed.
The inner factor names the same shared storage as the Krylov factor
(``empty_factor(..., storage=...)``), so a preconditioner held in shared
storage serves every program that names it under the inner backend's rules.

Solves differentiate through the phased primitives exactly as exact solves
do: the tangent is a Krylov solve of ``db - dA x`` with the same factor and
tolerance, and the cotangent a Krylov solve of the transposed operator,
preconditioned by the inner factor's transposed solve. The preconditioner is
never differentiated: the inner factor rides in the handle, which carries no
derivative.

[record_reports][diff_soup.linear._krylov.record_reports] is a diagnostic
that collects the status, iterations and residual of every Krylov solve on the
host.
"""

from __future__ import annotations

import contextlib
import functools
import threading
import types
from collections.abc import Iterator
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray
from jaxtyping import Array

from diff_soup._errors import ConfigurationError
from diff_soup._jax_compat import mlir
from diff_soup.linear._api import LinearSolver, SparseTopology, _bind_solve, analyze
from diff_soup.linear._factor import (
    Analysis,
    Factor,
    FactorContractError,
    Tolerance,
    check_tolerance,
    effective_transpose,
)
from diff_soup.linear._krylov._gmres import gmres_core
from diff_soup.linear._krylov._transposed import transposed_topology, transposed_values
from diff_soup.linear._operator import apply_assembled
from diff_soup.linear._primitive import (
    _split_solve_operands,
    pattern_matvec,
    phased_factor_p,
    phased_invalidate_p,
    phased_refactor_p,
)

__all__ = ["Krylov", "exact_analysis", "exact_backend", "record_reports", "report_outcome"]

#: The default stopping rule of a Krylov backend built without one.
DEFAULT_TOLERANCE = Tolerance(rtol=1e-10, atol=0.0, max_iter=100, restart=20)
#: The default cap of a solve on a possibly stale preconditioner.
DEFAULT_LAGGED_MAX_ITER = 10


class _KrylovPlan:
    """The inner analyses of one Krylov analysis, made on first use, per orientation."""

    __slots__ = ("inner", "lock")

    def __init__(self) -> None:
        """Start with no inner analysis."""
        self.inner: dict[str, Analysis] = {}
        self.lock = threading.Lock()


def _lanes(fn, batched: bool, *args):
    """``fn`` on unbatched arguments, or mapped over their shared leading axis."""
    return jax.vmap(fn)(*args) if batched else fn(*args)


class Krylov(LinearSolver):
    """Restarted GMRES, preconditioned by the factor of a direct backend.

    It solves float64 systems to a tolerance rather than exactly. A refactor always takes the
    new values, and its predicate decides only whether the preconditioner is factored again.
    Pass it as `linear=ds.linear.Krylov(ds.linear.CuDSS())`.

    Args:
        inner: The direct backend whose factor preconditions the iteration, for example
            `CuDSS()`.
        tol: The default stopping rule, `Tolerance(rtol=1e-10, atol=0.0, max_iter=100,
            restart=20)` when `None`.
        lagged_max_iter: The iteration cap of each linear solve inside the solvers, default
            `10`.
    """

    backend_name = "krylov"
    is_exact = False
    accepts_operator = True
    transpose_is_free = True
    supported_precisions = ("fp64",)
    refactor_keeps_values = False

    inner: LinearSolver = eqx.field(static=True)
    tol: Tolerance = eqx.field(static=True)
    lagged_max_iter: int = eqx.field(static=True)

    def __init__(
        self,
        inner: LinearSolver,
        tol: Optional[Tolerance] = None,
        lagged_max_iter: int = DEFAULT_LAGGED_MAX_ITER,
    ):
        """Hold the inner backend, the default tolerance and the lagged iteration cap."""
        if not isinstance(inner, LinearSolver):
            raise ConfigurationError(
                "Krylov needs inner= a configured diff_soup.linear.LinearSolver whose factor "
                f"preconditions the iteration, for example CuDSS() or Dense(); got "
                f"{type(inner).__name__}."
            )
        if not inner.is_exact:
            raise ConfigurationError(
                f"Krylov needs an exact inner backend (its factor is the preconditioner); "
                f"{type(inner).__name__} declares is_exact=False."
            )
        if type(inner)._analyze is LinearSolver._analyze:
            raise ConfigurationError(
                f"Krylov needs an inner backend that implements the analyze/factor/solve "
                f"contract; {type(inner).__name__} defines no _analyze hook."
            )
        if "fp64" not in tuple(inner.supported_precisions):
            raise ConfigurationError(
                f"Krylov solves at double precision and needs an inner backend that factors at "
                f"'fp64'; {type(inner).__name__} declares {tuple(inner.supported_precisions)}."
            )
        try:
            tol = check_tolerance(DEFAULT_TOLERANCE if tol is None else tol)
        except FactorContractError as exc:
            raise ConfigurationError(f"Krylov tol must be a valid Tolerance: {exc}") from exc
        if isinstance(lagged_max_iter, bool) or not isinstance(lagged_max_iter, (int, np.integer)):
            raise ConfigurationError(
                f"Krylov lagged_max_iter must be a positive integer; got {lagged_max_iter!r}."
            )
        if int(lagged_max_iter) < 1:
            raise ConfigurationError(
                f"Krylov lagged_max_iter must be a positive integer; got {lagged_max_iter!r}."
            )
        self.inner = inner
        self.tol = tol
        self.lagged_max_iter = int(lagged_max_iter)

    # ---- capabilities read through the instance ---------------------------------
    @property
    def default_tolerance(self) -> Tolerance:
        """The stopping rule of a solve that passes no ``tol``: this backend's ``tol``."""
        return self.tol

    @property
    def platforms(self) -> tuple:
        """The inner backend's platforms."""
        return tuple(self.inner.platforms)

    @property
    def factor_residency(self) -> str:
        """The inner backend's factor residency (where the preconditioner lives)."""
        return self.inner.factor_residency

    @property
    def vmap_factors(self) -> str:
        """The inner backend's mode of batching factors over values."""
        return self.inner.vmap_factors

    @property
    def supports_reordering(self) -> bool:
        """Whether the inner backend selects a fill-reducing ordering from the layout."""
        return bool(self.inner.supports_reordering)

    @property
    def supports_hybrid_memory(self) -> bool:
        """Whether the inner backend spills factor storage to host memory."""
        return bool(self.inner.supports_hybrid_memory)

    # ---- analysis ---------------------------------------------------------------
    def _analyze(
        self, topology: SparseTopology, options: tuple, layout: Any, precision: str
    ) -> _KrylovPlan:
        """Record nothing yet: the inner analyses are made on first use, per orientation.

        Raises:
            ConfigurationError: For a non-square topology.
        """
        del options, layout, precision
        if int(topology.n_dofs) < 1:
            raise ConfigurationError(
                f"Krylov needs a topology with at least one unknown; got n_dofs={topology.n_dofs}."
            )
        return _KrylovPlan()

    def _inner_analysis(self, analysis: Analysis, orientation: str) -> Analysis:
        """The inner backend's analysis of the operator a factor of ``orientation`` preconditions.

        Orientation ``"A"`` (and every orientation of an analysis declared
        symmetric) analyses the topology itself with the analysis's options,
        layout, precision and structure declaration; ``"AT"`` analyses the
        transposed pattern as a plain topology.

        Raises:
            FactorContractError: For orientation ``"AT"`` over an inner backend
                whose values describe more than the pattern.
        """
        plan = analysis.backend_analysis
        key = "A" if orientation == "A" or analysis.symmetric else "AT"
        found = plan.inner.get(key)
        if found is not None:
            return found
        if key == "AT" and getattr(self.inner, "pattern_matvec", None) is not None:
            raise FactorContractError(
                f"Krylov over {type(self.inner).__name__} factors orientation 'A' only: the "
                "inner backend's values describe more than the pattern, so the transposed "
                "pattern cannot carry them. Factor orientation 'A' and solve with "
                "transpose=True."
            )
        if key == "A":
            topology, symmetric, spd = analysis.topology, analysis.symmetric, analysis.spd
        else:
            topology, symmetric, spd = transposed_topology(analysis.topology)[0], False, False
        built = analyze(
            topology,
            backend=self.inner,
            options=analysis.options,
            layout=analysis.layout,
            precision=analysis.precision,
            symmetric=symmetric,
            spd=spd,
        )
        with plan.lock:
            return plan.inner.setdefault(key, built)

    def _inner_values(self, analysis: Analysis, orientation: str, values: Array) -> Array:
        """The values of the operator the inner factor of ``orientation`` factors."""
        if orientation == "A" or analysis.symmetric:
            return values
        return transposed_values(analysis.topology, values)

    @staticmethod
    def _held(
        inner: Analysis, handle: tuple, precision: str, storage: Optional[str], order: Array
    ) -> Factor:
        """The inner factor a Krylov handle holds (values, handle leaves, generation), at ``order``."""
        return Factor(
            analysis=inner,
            orientation="A",
            precision=precision,
            exact=True,
            values=handle[0],
            handle=tuple(handle[1:-1]),
            generation=handle[-1],
            order=order,
            storage=storage,
        )

    # ---- contract hooks ---------------------------------------------------------
    def factor_avals(
        self, analysis: Analysis, values_aval: Any, *, orientation: str, precision: str, tol: Any
    ) -> tuple:
        """The inner factor's values, its handle leaves and its ``int32`` generation."""
        del tol
        inner = self._inner_analysis(analysis, orientation)
        lane = ShapedArray((inner.nnz,), values_aval.dtype)
        handle = inner.backend.factor_avals(
            inner, lane, orientation="A", precision=precision, tol=None
        )
        return (lane, *handle, ShapedArray((), jnp.int32))

    def empty_handle(
        self,
        analysis: Analysis,
        values_aval: Any,
        *,
        context_id: int,
        orientation: str,
        precision: str,
        tol: Any,
    ) -> tuple:
        """The inner backend's empty handle with zero values and generation ``-1``."""
        del tol
        inner = self._inner_analysis(analysis, orientation)
        lane = ShapedArray((inner.nnz,), values_aval.dtype)
        handle = inner.backend.empty_handle(
            inner, lane, context_id=context_id, orientation="A", precision=precision, tol=None
        )
        return (jnp.zeros(lane.shape, lane.dtype), *handle, jnp.full((), -1, jnp.int32))

    def factor_fun(
        self,
        values: Array,
        token: Array,
        context_id: Array,
        *,
        order: Array,
        analysis: Analysis,
        orientation: str,
        precision: str,
        tol: Any,
    ) -> tuple:
        """Factor the preconditioner from the operator's own values (one inner factorization).

        The inner factorization takes the order word and its order word is
        returned.

        Returns:
            ``(inner_values, *inner_handle, generation, status, order)``.
        """
        del tol
        inner = self._inner_analysis(analysis, orientation)
        fresh = self._inner_values(analysis, orientation, values)

        def one(v, o):
            *handle, status, o = phased_factor_p.bind(
                v,
                token,
                context_id,
                o,
                analysis=inner,
                orientation="A",
                precision=precision,
                tol=None,
            )
            return (v, *handle, jnp.zeros((), jnp.int32), status, o)

        return _lanes(one, values.ndim == 2, fresh, order)

    def apply_fun(
        self,
        values: Array,
        *handle: Array,
        order: Array,
        b: Array,
        analysis: Analysis,
        orientation: str,
        transpose: bool,
        precision: str,
        tol: Any,
        weights: Optional[Array] = None,
        x0: Optional[Array] = None,
        storage: Optional[str] = None,
    ) -> tuple:
        """GMRES on the operator of ``values``, preconditioned by the held inner factor.

        The operator is ``A_bc`` or ``A_bc^T`` of ``values`` as the
        orientation and ``transpose`` select; the preconditioner is a solve
        with the inner factor, transposed when ``transpose`` is. A block of
        right-hand sides runs its columns side by side; the counts and the
        status of a lane are those of its slowest column (the largest status
        code). ``storage`` is the shared storage the inner factor names. Every
        preconditioner application is an inner solve that takes the order
        word of the one before it (the first takes ``order``), carried through
        the iteration, and the last one's order word is returned.

        Returns:
            ``(x, status, iterations, applications, order)``.
        """
        inner = self._inner_analysis(analysis, orientation)
        held = self._held(inner, handle, precision, storage, order)
        flip = effective_transpose(orientation, transpose)
        batched = values.ndim == 2

        def mv(x: Array) -> Array:
            return apply_assembled(analysis, values, x, transpose=flip)

        def one(f: Factor, word: Array, rhs: Array) -> tuple:
            x, status, *_, solved = _bind_solve(
                Factor(
                    analysis=f.analysis,
                    orientation=f.orientation,
                    precision=f.precision,
                    exact=f.exact,
                    values=f.values,
                    handle=f.handle,
                    generation=f.generation,
                    order=word,
                    storage=f.storage,
                ),
                rhs,
                transpose=transpose,
                tol=None,
                weights=None,
                x0=None,
                caller="solve",
            )
            return x, status, solved.order

        def precondition(v: Array, word: Array) -> tuple:
            return _lanes(one, batched, held, word, v)

        x, status, iterations, applications, residual, order = gmres_core(
            mv, b, precondition=precondition, tol=tol, weights=weights, x0=x0, order=order
        )
        if b.ndim - (values.ndim - 1) == 2:
            status, iterations = jnp.max(status, axis=-1), jnp.max(iterations, axis=-1)
            applications, residual = jnp.max(applications, axis=-1), jnp.max(residual, axis=-1)
        if _RECORDERS:
            _emit(tuple(_RECORDERS), orientation, bool(transpose), status, iterations, residual)
        return x, status, iterations, applications, order

    def refactor_fun(
        self,
        values: Array,
        held_values: Array,
        predicate: Array,
        *handle: Array,
        order: Array,
        analysis: Analysis,
        orientation: str,
        precision: str,
        tol: Any,
        adopt: bool,
        storage: Optional[str] = None,
    ) -> tuple:
        """Take ``values`` as the operator in every lane; rebuild the preconditioner where asked.

        The inner factor is refactored from the inner values of ``values``
        where ``predicate`` holds, in place in the storage it names (the
        inner refactor runs with ``adopt=True`` whatever ``adopt`` is) and in
        the shared storage ``storage`` when the factor names one; elsewhere it
        is kept (and a shared storage made to hold it, by the inner backend's
        rule). The held operator values are not used. The inner refactor takes
        the order word and its order word is returned.

        Returns:
            ``(values, inner_values, *inner_handle, generation, status, order)``.
        """
        del held_values, tol, adopt
        inner = self._inner_analysis(analysis, orientation)
        fresh = self._inner_values(analysis, orientation, values)
        n_inner = len(handle) - 2

        def one(v, p, held, generation, o, *inner_handle):
            kept, *new_handle, status, o = phased_refactor_p.bind(
                v,
                held,
                p,
                *inner_handle,
                o,
                n_handle=n_inner,
                analysis=inner,
                orientation="A",
                precision=precision,
                tol=None,
                adopt=True,
                storage=storage,
            )
            generation = jnp.where(p, generation + 1, generation).astype(jnp.int32)
            return (kept, *new_handle, generation, status, o)

        predicate = jnp.asarray(predicate, jnp.bool_)
        outs = _lanes(
            one,
            values.ndim == 2,
            fresh,
            predicate,
            handle[0],
            handle[-1],
            order,
            *handle[1:-1],
        )
        kept, *new_handle, generation, status, order = outs
        return (values, kept, *new_handle, generation, status, order)

    # ---- lowerings that pass the shared-storage name to the inner factor ----------
    def lower_apply(self, ctx: Any, *args: Any, **params: Any) -> Any:
        """Lower a Krylov solve as its JAX-level function, with the factor's storage name."""
        storage = params.pop("storage", None)
        n_handle = params.pop("n_handle")
        has_weights = bool(params.pop("has_weights", False))
        has_x0 = bool(params.pop("has_x0", False))
        result = ctx.avals_out[0].dtype

        def run(values, *operands):
            handle, order, b, weights, x0 = _split_solve_operands(
                "phased_solve", operands, n_handle, has_weights, has_x0
            )
            x, status, iterations, applications, order = self.apply_fun(
                values,
                *handle,
                order=order,
                b=b,
                weights=weights,
                x0=x0,
                storage=storage,
                **params,
            )
            return [
                x.astype(result),
                status.astype(jnp.int32),
                iterations.astype(jnp.int32),
                applications.astype(jnp.int32),
                order.astype(jnp.int32),
            ]

        return mlir.lower_fun(run, multiple_results=True)(ctx, *args)

    def lower_refactor(self, ctx: Any, *args: Any, **params: Any) -> Any:
        """Lower a Krylov refactor as its JAX-level function, with the factor's storage name."""
        storage = params.pop("storage", None)
        params.pop("n_handle")
        dtypes = [aval.dtype for aval in ctx.avals_out]

        def run(values, held_values, predicate, *operands):
            *handle, order = operands
            outs = self.refactor_fun(
                values, held_values, predicate, *handle, order=order, storage=storage, **params
            )
            return [jnp.asarray(out).astype(dtype) for out, dtype in zip(outs, dtypes)]

        return mlir.lower_fun(run, multiple_results=True)(ctx, *args)

    def lower_invalidate(self, ctx: Any, *args: Any, **params: Any) -> Any:
        """Lower an invalidation of a Krylov factor as the inner factor's invalidation."""
        analysis = params["analysis"]
        inner = self._inner_analysis(analysis, params["orientation"])
        storage = params.get("storage")

        def run(*operands):
            *handle, context_id, order = operands
            inner_handle = tuple(handle[1:-1])
            return list(
                phased_invalidate_p.bind(
                    *inner_handle,
                    context_id,
                    order,
                    n_handle=len(inner_handle),
                    analysis=inner,
                    orientation="A",
                    precision=params["precision"],
                    storage=storage,
                )
            )

        return mlir.lower_fun(run, multiple_results=True)(ctx, *args)

    # ---- hooks forwarded to the inner backend -------------------------------------
    def label_empty_lanes(self, handle: tuple, lanes: Any, size: int) -> tuple:
        """Relabel the inner handle's empty lanes with the inner backend's hook, if it has one."""
        label = getattr(self.inner, "label_empty_lanes", None)
        if label is None:
            return handle
        inner_handle = label(tuple(handle[1:-1]), lanes, size)
        return (handle[0], *inner_handle, handle[-1])

    def pattern_matvec(self, analysis: Analysis, dvalues: Any, x: Any, *, transpose: bool) -> Array:
        """The pattern matvec of the operator, through the inner backend's when it has one.

        The family's derivative rules and residuals read the operator through
        this method; an inner backend whose values describe more than the
        pattern (a low-rank term) supplies the whole operator.
        """
        if getattr(self.inner, "pattern_matvec", None) is not None:
            inner = self._inner_analysis(analysis, "A")
            return pattern_matvec(inner, dvalues, x, transpose=transpose)
        plain = types.SimpleNamespace(topology=analysis.topology)
        return pattern_matvec(plain, dvalues, x, transpose=transpose)

    def declare_storage(self, analysis: Analysis, context_id: int, storage: str) -> None:
        """Declare the shared storage on every inner analysis made so far, as the inner backend does."""
        declare = getattr(self.inner, "declare_storage", None)
        if declare is None:
            return
        for inner in tuple(analysis.backend_analysis.inner.values()):
            declare(inner, context_id, storage)

    def _release(self, ctx: Any) -> None:
        """Release the inner backend's storage held for ``ctx``."""
        self.inner._release(ctx)

    def _resident_bytes(self, ctx: Any) -> int:
        """The inner backend's storage held for ``ctx``, counted once per backend that holds it."""
        holder = _storage_holder(self)
        for registered in tuple(getattr(ctx, "_backends", ())):
            if _storage_holder(registered) is holder:
                if registered is not self:
                    return 0
                break
        return int(self.inner._resident_bytes(ctx))


def _storage_holder(backend: Any) -> type:
    """The class of the backend whose native storage a (wrapping) backend uses."""
    while isinstance(getattr(backend, "inner", None), LinearSolver):
        backend = backend.inner
    return type(backend)


def exact_backend(backend: Any) -> Any:
    """The exact backend behind ``backend``: a Krylov backend's ``inner``, else ``backend``.

    Consumers that need the exact inverse of their matrix (the forward
    sensitivity solves, the consistent initialisation, the projections) solve
    with it when the caller configured a Krylov backend.

    Args:
        backend: A backend instance, a backend name or ``None``.

    Returns:
        ``backend.inner`` for a [Krylov][diff_soup.linear.Krylov] backend,
        else ``backend`` unchanged.
    """
    return backend.inner if isinstance(backend, Krylov) else backend


def exact_analysis(analysis: Analysis) -> Analysis:
    """The inner backend's analysis of the same pattern for a Krylov analysis, else ``analysis``.

    Args:
        analysis: A contract [Analysis][diff_soup.linear.Analysis].

    Returns:
        The analysis an exact solve of the same operator uses.
    """
    backend = analysis.backend
    if isinstance(backend, Krylov):
        return backend._inner_analysis(analysis, "A")
    return analysis


# --------------------------------------------------------------------------- #
# Diagnostic: the reports of every Krylov solve, on the host
# --------------------------------------------------------------------------- #
_RECORDERS: list = []
_RECORDERS_LOCK = threading.Lock()


@contextlib.contextmanager
def record_reports() -> Iterator[list]:
    """Collect the outcome of every Krylov solve of the programs lowered inside the context.

    A diagnostic, not a counter: while the context is active, every Krylov
    solve that is traced and lowered (a jitted function compiled inside the
    context, or an eager solve) gets a host callback (``jax.debug.callback``,
    unordered) that appends one record per execution to the yielded list,
    also after the context exits for programs compiled inside it. Programs
    compiled outside the context record nothing and carry no callback. Each
    record is a dict with ``role`` (``"solve"``), ``orientation`` and
    ``transpose`` (the factor's orientation and the solve's flag) and the
    per-lane NumPy arrays ``status``, ``iterations`` and ``residual`` (a
    block's slowest column). A consumer that retries a solve on a rebuilt
    preconditioner may add one record of its final outcome with a role of its
    own and a ``rebuilt`` array (see
    [report_outcome][diff_soup.linear._krylov.report_outcome]). Callbacks run
    asynchronously: call ``jax.effects_barrier()`` before reading the list.

    Yields:
        The list the records are appended to.
    """
    records: list = []
    with _RECORDERS_LOCK:
        _RECORDERS.append(records)
    try:
        yield records
    finally:
        with _RECORDERS_LOCK:
            _RECORDERS[:] = [kept for kept in _RECORDERS if kept is not records]


def report_outcome(
    role: str,
    orientation: str,
    transpose: bool,
    status: Any,
    iterations: Any,
    residual: Any,
    rebuilt: Any = None,
) -> None:
    """Add one record of a consumer's final outcome to the active ``record_reports`` lists.

    Traced at the consumer's call site and a no-op (nothing traced) when no
    ``record_reports`` context is active; see
    [record_reports][diff_soup.linear._krylov.record_reports].

    Args:
        role: What the outcome is, for example ``"cotangent"``.
        orientation: The orientation of the factor solved with.
        transpose: Whether the solve was transposed.
        status: The final ``int32`` status per lane.
        iterations: The iterations per lane, all attempts together.
        residual: The final residual per lane.
        rebuilt: ``None``, or a boolean per lane: whether the preconditioner
            was rebuilt.
    """
    if not _RECORDERS:
        return
    extras = () if rebuilt is None else (rebuilt,)
    jax.debug.callback(
        functools.partial(_append, tuple(_RECORDERS), role, orientation, bool(transpose)),
        status,
        iterations,
        residual,
        *extras,
        ordered=False,
    )


def _append(
    targets: tuple,
    role: str,
    orientation: str,
    transpose: bool,
    status,
    iterations,
    residual,
    rebuilt=None,
):
    """Host side of the diagnostic: append one record to every target list."""
    record = {
        "role": role,
        "orientation": orientation,
        "transpose": transpose,
        "status": np.asarray(status),
        "iterations": np.asarray(iterations),
        "residual": np.asarray(residual),
    }
    if rebuilt is not None:
        record["rebuilt"] = np.asarray(rebuilt)
    for target in targets:
        target.append(record)


def _emit(targets: tuple, orientation: str, transpose: bool, status, iterations, residual) -> None:
    """Trace the host callback of the diagnostic for one Krylov solve."""
    jax.debug.callback(
        functools.partial(_append, targets, "solve", orientation, transpose),
        status,
        iterations,
        residual,
        ordered=False,
    )
