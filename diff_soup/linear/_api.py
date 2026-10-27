"""The analyze / factor / solve contract and the abstract backend class.

The module-level functions [analyze][diff_soup.linear.analyze],
[factor][diff_soup.linear.factor], [empty_factor][diff_soup.linear.empty_factor],
[refactor][diff_soup.linear.refactor], [solve][diff_soup.linear.solve],
[solve_checked][diff_soup.linear.solve_checked] and
[invalidate][diff_soup.linear.invalidate] work over any backend that
implements the contract hooks of
[LinearSolver][diff_soup.linear._api.LinearSolver] (``Dense``, ``KLU``,
``SuperLU``, ``CuDSS``). ``analyze`` is a host operation memoised by content;
the others are traced and compose with ``jit``, ``vmap`` and both autodiff
modes. Every operation on a factor returns the factor to use next, and
[after][diff_soup.linear.after] links the order words of factors that name
one storage.

ClassVar capability flags on each backend are checked eagerly at ``analyze``
time so unsupported modes raise [ConfigurationError][diff_soup._errors.ConfigurationError]
immediately rather than silently falling back.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray
from jaxtyping import Array, ArrayLike

from diff_soup._errors import ConfigurationError
from diff_soup._layout import Layout, linear_mode
from diff_soup._precision import Precision
from diff_soup.linear._factor import (
    FACTOR_RESIDENCIES,
    PRECISION_RUNGS,
    VMAP_FACTOR_MODES,
    Analysis,
    Factor,
    FactorContractError,
    SolveContext,
    SolveReport,
    Tolerance,
    check_orientation,
    check_tolerance,
    dirichlet_rows,
    effective_transpose,
    rung_dtype,
    solve_dtype,
)
from diff_soup.linear._operator import apply_assembled
from diff_soup.linear._primitive import (
    _compiled,
    check_empty_handle,
    phased_factor_p,
    phased_invalidate_p,
    phased_refactor_p,
    phased_solve_p,
)


def _freeze_options(options: Optional[dict]) -> tuple[tuple[str, Any], ...]:
    """Normalize a user options dict to a hashable, sorted tuple of items.

    ``options`` is recorded on an [Analysis][diff_soup.linear.Analysis] and
    keys the memo of [analyze][diff_soup.linear.analyze], so it must be
    hashable. Any user dict (mutable, unordered, unhashable as a key) is
    normalized once at ``analyze`` entry to a tuple of ``(key, value)`` pairs
    sorted by key.

    Args:
        options: User options dict, or ``None``.

    Returns:
        A sorted tuple of ``(key, value)`` items (empty for ``None``).
    """
    if not options:
        return ()
    return tuple(sorted(options.items(), key=lambda kv: kv[0]))


def _options_dict(options: Any) -> dict[str, Any]:
    """Re-expand a frozen options tuple (or dict/None) back to a plain dict.

    The inverse of [diff_soup.linear._api._freeze_options][], tolerant of the forms a
    backend may receive: the frozen tuple recorded on an
    [Analysis][diff_soup.linear.Analysis], a raw dict, a dataclass-based
    typed option carrier, or ``None``.

    Args:
        options: A frozen ``((k, v), ...)`` tuple, a dict, a dataclass instance,
            or ``None``.

    Returns:
        A fresh plain dict.
    """
    if options is None:
        return {}
    if isinstance(options, dict):
        return dict(options)
    if dataclasses.is_dataclass(options):
        return {
            field.name: getattr(options, field.name)
            for field in dataclasses.fields(options)
            if not field.name.startswith("_")
        }
    return dict(options)


def option_field_items(backend: Any) -> tuple[tuple[str, Any], ...]:
    """The ``(name, value)`` pairs of ``backend``'s settings, without those left out.

    A backend lists its settings, the static fields that are analyze
    options, in ``option_fields``, and validates these pairs with its option
    parser when it is built, so a bad setting fails at construction rather
    than at the first solve. A setting whose default is ``None`` is left out
    at ``None``.

    Raises:
        ConfigurationError: For a setting whose value is not hashable.
    """
    defaults = {field.name: field.default for field in dataclasses.fields(backend)}
    items = []
    for name in type(backend).option_fields:
        value = getattr(backend, name)
        if value is None and defaults[name] is None:
            continue
        try:
            hash(value)
        except TypeError as exc:
            raise ConfigurationError(
                f"{public_backend_name(backend)} setting {name} must be hashable; got {value!r}."
            ) from exc
        items.append((name, value))
    return tuple(items)


def carried_options(backend: Any) -> dict[str, Any]:
    """The analyze options ``backend`` carries: its settings that differ from their defaults.

    A setting at its default is left out, so a backend built without
    arguments carries no options and analyzes exactly as the name that
    selects it.
    [analyze][diff_soup.linear.analyze] merges these under its ``options=``,
    whose keys win.
    """
    names = type(backend).option_fields
    if not names:
        return {}
    defaults = {field.name: field.default for field in dataclasses.fields(backend)}
    carried = {}
    for name in names:
        value = getattr(backend, name)
        default = defaults[name]
        if value is default:
            continue
        try:
            same = default is not None and bool(value == default)
        except (TypeError, ValueError):
            same = False
        if not same:
            carried[name] = value
    return carried


def _to_int32_bytes(arr: Any) -> tuple[bytes, int]:
    """Normalize a host index array to ``(contiguous int32 bytes, length)``."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.int32).reshape(-1))
    return a.tobytes(), int(a.shape[0])


class SparseTopology(eqx.Module):
    """The sparsity pattern of a square matrix, given by the coordinates of its entries.

    The values passed to `factor` follow the order of `rows` and `cols`, one per entry, and
    duplicate coordinates add. A Dirichlet row becomes an identity row, so the pattern must
    hold its diagonal entry `(i, i)`.

    Args:
        rows: The row index of each entry.
        cols: The column index of each entry.
        n_dofs: The number of rows and columns.
        bc_indices: The Dirichlet rows, or `None` for none.
    """

    n_dofs: int = eqx.field(static=True, converter=int)
    _rows_bytes: bytes = eqx.field(static=True)
    _cols_bytes: bytes = eqx.field(static=True)
    _nnz: int = eqx.field(static=True)
    _bc_bytes: Optional[bytes] = eqx.field(static=True, default=None)

    def __init__(
        self,
        rows: Any,
        cols: Any,
        n_dofs: int,
        bc_indices: Any = None,
    ):
        """Build the static pattern from host index arrays.

        Args:
            rows: Row indices (COO), coerced to contiguous int32 host bytes.
            cols: Column indices (COO), coerced to contiguous int32 host bytes.
            n_dofs: Matrix side length.
            bc_indices: Optional Dirichlet row indices, or ``None``.
        """
        self.n_dofs = int(n_dofs)
        self._rows_bytes, self._nnz = _to_int32_bytes(rows)
        self._cols_bytes, _ = _to_int32_bytes(cols)
        self._bc_bytes = None if bc_indices is None else _to_int32_bytes(bc_indices)[0]

    @classmethod
    def full(cls, n: int) -> "SparseTopology":
        """The pattern of a dense `n` by `n` matrix, with its entries in row-major order.

        Args:
            n: The number of rows and columns.

        Returns:
            The `SparseTopology`.
        """
        idx = np.arange(int(n), dtype=np.int32)
        return cls(rows=np.repeat(idx, int(n)), cols=np.tile(idx, int(n)), n_dofs=int(n))

    @property
    def rows(self) -> np.ndarray:
        """The row index of each entry, as a read-only int32 NumPy array."""
        return np.frombuffer(self._rows_bytes, dtype=np.int32)

    @property
    def cols(self) -> np.ndarray:
        """The column index of each entry, as a read-only int32 NumPy array."""
        return np.frombuffer(self._cols_bytes, dtype=np.int32)

    @property
    def bc_indices(self) -> Optional[np.ndarray]:
        """The Dirichlet rows as a read-only int32 NumPy array, or `None`."""
        if self._bc_bytes is None:
            return None
        return np.frombuffer(self._bc_bytes, dtype=np.int32)

    def __hash__(self) -> int:
        """Hash by ``(n_dofs, rows bytes, cols bytes)`` for static jit use."""
        return hash((self.n_dofs, self._rows_bytes, self._cols_bytes))

    def __eq__(self, other: object) -> bool:
        """Structural equality over ``n_dofs`` + rows / cols / bc bytes."""
        if not isinstance(other, SparseTopology):
            return NotImplemented
        return (
            self.n_dofs == other.n_dofs
            and self._rows_bytes == other._rows_bytes
            and self._cols_bytes == other._cols_bytes
            and self._bc_bytes == other._bc_bytes
        )


#: The precision rungs a solve mode of a float64 problem factors at. A
#: reduced-precision mode factors at ``"fp32"`` and falls back to, or
#: switches to, an ``"fp64"`` factor of the same pattern.
_SOLVE_MODE_RUNGS: dict[str, tuple[str, ...]] = {
    "fp64": ("fp64",),
    "fp32": ("fp32", "fp64"),
    "fp32_ir": ("fp32", "fp64"),
    "adaptive": ("fp32", "fp64"),
}


def solve_modes_of(supported_precisions: Sequence[str]) -> frozenset[str]:
    """The ``Precision.solve_mode`` values a backend with these rungs serves for float64 values.

    A mode is served when the backend factors at every rung the mode uses:
    ``"fp64"`` needs the ``"fp64"`` rung, and the reduced-precision modes
    (``"fp32"``, ``"fp32_ir"``, ``"adaptive"``) need both ``"fp32"`` and
    ``"fp64"``.

    Args:
        supported_precisions: The backend's ``supported_precisions``.

    Returns:
        The served solve modes.
    """
    rungs = set(supported_precisions)
    return frozenset(mode for mode, need in _SOLVE_MODE_RUNGS.items() if rungs.issuperset(need))


def precision_support_error(backend: Any, name: str, precision: Precision) -> Optional[str]:
    """Why ``backend`` cannot serve ``precision``, or ``None`` when it can.

    Answers from the backend's ``supported_precisions``, the rungs
    [analyze][diff_soup.linear.analyze] accepts. A single-precision compute
    dtype needs the ``"fp32"`` rung (the values are factored without
    promotion); a float64 problem needs the rungs of its solve mode (see
    [solve_modes_of][diff_soup.linear._api.solve_modes_of]).

    Args:
        backend: The backend class or instance.
        name: The backend name used in the message.
        precision: The requested [Precision][diff_soup._precision.Precision].

    Returns:
        The error message, or ``None``.
    """
    rungs = tuple(backend.supported_precisions)
    if np.dtype(precision.compute_dtype).itemsize <= 4:
        if "fp32" in rungs:
            return None
        return (
            f"the {name} linear backend factors in double precision only, so it "
            f"cannot solve a {np.dtype(precision.compute_dtype)} problem without "
            "promoting it. Use linear='dense' (CPU or GPU) or linear='cudss' (GPU), "
            "or build the problem with a float64 Precision."
        )
    supported = solve_modes_of(rungs)
    if precision.solve_mode in supported:
        return None
    return (
        f"the {name} linear backend does not support "
        f"Precision.solve_mode={precision.solve_mode!r}; supported modes: "
        f"{sorted(supported)}. Select a backend that factors at both 'fp64' and 'fp32' "
        "(linear='dense' or linear='cudss') for a reduced-precision factor of a float64 "
        "problem, or keep the default fp64 precision."
    )


def public_backend_name(backend: Any) -> str:
    """The name ``diff_soup.linear`` exports for ``backend``, with the backend it wraps.

    ``KLU`` for ``KLU()``, ``Woodbury over KLU`` for a ``Woodbury`` whose
    inner backend is ``KLU()``, ``Krylov over CuDSS`` for ``Krylov(CuDSS())``;
    a class the package does not export keeps its own name. Refusals name
    backends this way, as users spell them.
    """
    import diff_soup.linear as linear

    exported = {}
    for export in linear.__all__:
        value = getattr(linear, export, None)
        if isinstance(value, type) and export != "LinearSolver":
            exported[id(value)] = export
    cls = type(backend)
    name = next((exported[id(c)] for c in cls.__mro__ if id(c) in exported), cls.__name__)
    inner = getattr(backend, "inner", None)
    if isinstance(inner, LinearSolver):
        return f"{name} over {public_backend_name(inner)}"
    return name


class LinearSolver(eqx.Module):
    """The base class of the linear backends.

    Subclass it only to write a backend of your own. Its class attributes declare what the
    backend supports, and `analyze` refuses a request outside them.
    """

    # ---- identity -----------------------------------------------------------
    # The concrete backend name ("dense" / "klu" / "superlu" / "cudss") — the
    # same name ``resolve_backend`` accepts and the name a solve's
    # ``SolverStats`` reports as the realized linear backend. ``None`` on the
    # abstract base.
    backend_name: ClassVar[Optional[str]] = None
    """The backend's name, such as `"klu"` or `"dense"`, or `None` on the base class."""
    # The backend's settings: the names of its static fields that are analyze
    # options (``KLU(ordering="colamd")``). ``analyze`` merges each one set
    # away from its default under an explicit ``options=``, whose keys win; a
    # field left at its default (``None`` where that means "left out") adds
    # nothing, so ``KLU()`` analyzes exactly as ``linear="klu"``.
    option_fields: ClassVar[tuple[str, ...]] = ()

    # ---- capabilities (ClassVar; eager-checked at composition time) --------
    # The JAX platforms (``"cpu"``, ``"cuda"``, ...) whose programs the backend
    # can run; empty means every platform. A program lowered for a platform
    # outside the list is refused with a ``FactorContractError``.
    platforms: ClassVar[tuple[str, ...]] = ()
    """The JAX platforms the backend runs on, such as `("cpu",)`, where empty means every platform."""
    # Layout capabilities, read by ``_validate_capabilities``. Selection of
    # the fill-reducing ordering via ``layout.reorder``: ``False`` when the
    # backend has no ordering step, or has a fixed one it cannot be told to
    # change; such a backend rejects a non-default request rather than
    # accepting it and quietly factorizing in a different order than asked.
    supports_reordering: ClassVar[bool] = False
    # Host-memory spill of factor storage (``layout.hybrid_memory``).
    supports_hybrid_memory: ClassVar[bool] = False
    # One factorization spread over the devices of a ``"model"`` mesh axis:
    # across processes (``"mgmn"``) or across the local devices of one
    # process (``"mg"``). Only the cuDSS backend declares them.
    supports_mgmn: ClassVar[bool] = False
    supports_mg_direct: ClassVar[bool] = False

    # ---- analyze / factor / solve contract ----------------------------------
    # Capabilities the module-level ``analyze`` / ``factor`` / ``solve`` /
    # ``refactor`` functions read. Consumers may rely on these flags and on
    # nothing else about a backend's factors.
    #
    # Where factor storage lives: ``"arrays"`` (the ``Factor.handle`` arrays
    # are the whole factorization), ``"device"`` or ``"host"`` (native
    # storage owned by a ``SolveContext``; ``factor`` then needs a context,
    # and rematerialisation saves the outputs of its solves as it saves a
    # factorization's, since native storage may have changed by the time a
    # recomputed solve would run).
    factor_residency: ClassVar[str] = "arrays"
    """Where a factor is stored, `"arrays"` in the factor itself or `"host"` or `"device"` memory a `SolveContext` owns."""
    # Whether one factor serves both ``transpose=False`` and
    # ``transpose=True`` solves without another factorization. ``False``
    # unless the backend declares it: a consumer must never assume a free
    # transposed solve the backend does not promise.
    transpose_is_free: ClassVar[bool] = False
    """Whether one factor serves `transpose=True` solves without another factorization."""
    # How a ``vmap`` over operator values runs: ``"native"`` (one batched
    # factorization), ``"sequential"`` (lanes mapped with ``lax.map``) or
    # ``"unsupported"`` (a typed error at trace time).
    vmap_factors: ClassVar[str] = "unsupported"
    """How `jax.vmap` over the values runs, `"native"` as one batch, `"sequential"` one matrix at a time, or `"unsupported"`."""
    # The precision rungs ``analyze`` accepts for this backend.
    supported_precisions: ClassVar[tuple[str, ...]] = ("fp64",)
    """The factor precisions `analyze` accepts, from `"fp64"` and `"fp32"`."""
    # Whether solves apply the exact inverse of the factored operator.
    is_exact: ClassVar[bool] = True
    """Whether a solve applies the exact inverse rather than iterating to a tolerance."""
    # The stopping rule an inexact backend's solve uses when the caller passes
    # no ``tol``; ``None`` means every solve must pass one. Exact backends
    # ignore tolerances.
    default_tolerance: ClassVar[Optional[Tolerance]] = None
    # Whether the backend takes a matrix-free operator instead of assembled
    # values (iterative backends).
    accepts_operator: ClassVar[bool] = False
    """Whether the backend works on a matrix-free operator rather than assembled values."""
    # Whether a refactor whose predicate is false keeps the held values (the
    # factor's operator stays the matrix it was built from). ``False`` for a
    # backend whose factor is a preconditioner of a current operator: its
    # refactor takes the new values in every lane and gates only the
    # preconditioner's rebuild, and the tangent of the kept values is the
    # tangent of the new values.
    refactor_keeps_values: ClassVar[bool] = True
    # MLIR lowerings of the factor / solve / refactor primitives (native
    # backends: FFI custom calls that take the order word as an operand and
    # return it as a result). ``None`` lowers the JAX-level
    # ``factor_fun`` / ``apply_fun`` / ``refactor_fun`` instead.
    lower_factor: ClassVar[Optional[Callable[..., Any]]] = None
    lower_apply: ClassVar[Optional[Callable[..., Any]]] = None
    lower_refactor: ClassVar[Optional[Callable[..., Any]]] = None
    # ``label_empty_lanes(handle, lanes, size) -> handle`` (optional, for
    # ``vmap_factors = "sequential"``): give every empty lane of a batch of
    # ``size`` handles a marker of its own before a refactor maps over the
    # lanes, so lanes that ask are factored into storage of their own.
    # ``None`` leaves the handles as they are.
    label_empty_lanes: ClassVar[Optional[Callable[..., Any]]] = None

    def _validate_capabilities(self, layout: Layout, precision: Precision | str) -> None:
        """Eagerly reject a ``layout`` / ``precision`` this backend cannot serve.

        Checks the precision against ``supported_precisions`` (a rung string
        must be one of them; a [Precision][diff_soup._precision.Precision]
        is checked by
        [precision_support_error][diff_soup.linear._api.precision_support_error]),
        ``layout.hybrid_memory`` against ``supports_hybrid_memory``,
        ``layout.reorder`` against ``supports_reordering``, and the
        ``"model"`` axis against ``supports_mgmn`` / ``supports_mg_direct``.

        Args:
            layout: The requested device-placement policy.
            precision: The requested precision rung (``"fp64"`` or
                ``"fp32"``) or dtype policy.

        Raises:
            ConfigurationError: If a requested mode is unsupported.
        """
        name = public_backend_name(self)
        rungs = tuple(self.supported_precisions)
        if isinstance(precision, str):
            if precision not in rungs:
                raise ConfigurationError(
                    f"{name} factors at {rungs} only, so precision={precision!r} is "
                    "unsupported; pass one of those rungs or use a backend that lists "
                    f"{precision!r} in supported_precisions."
                )
        else:
            problem = precision_support_error(self, name, precision)
            if problem is not None:
                raise ConfigurationError(problem)
        if layout.hybrid_memory and not self.supports_hybrid_memory:
            raise ConfigurationError(
                f"{name} does not support layout.hybrid_memory=True "
                "(host-memory spill). Use a backend that declares "
                "supports_hybrid_memory or set hybrid_memory=False."
            )
        if layout.reorder != "amd" and not self.supports_reordering:
            raise ConfigurationError(
                f"{name} cannot select a fill-reducing ordering, so "
                f"layout.reorder={layout.reorder!r} would not be applied. Use a "
                "backend that declares supports_reordering or leave "
                "reorder='amd'."
            )
        mode = linear_mode(layout)
        if mode == "mgmn" and not self.supports_mgmn:
            within = "stays in one process" if self.supports_mg_direct else "has one device"
            raise ConfigurationError(
                f"{name} cannot factor across processes on the model axis; use CuDSS, "
                f"or a layout whose model axis {within}."
            )
        if mode == "mg" and not self.supports_mg_direct:
            raise ConfigurationError(
                f"{name} cannot factor across local devices on the model axis; use CuDSS."
            )

    # ---- analyze / factor / solve contract hooks -----------------------------
    # The module-level ``analyze`` / ``factor`` / ``solve`` / ``refactor``
    # functions call these through ``analysis.backend``. A backend supplies
    # the symbolic analysis and, for each primitive, either a JAX-level
    # implementation (``factor_fun`` / ``apply_fun`` / ``refactor_fun``,
    # lowered with ``mlir.lower_fun`` and evaluated directly outside ``jit``)
    # or an MLIR lowering (``lower_factor`` / ``lower_apply`` /
    # ``lower_refactor``, called as ``hook(ctx, *operands, **params)``; eager
    # calls compile a one-equation program). The MLIR route is the one that
    # can make a native call take the order word as an operand. A backend
    # registers no derivative, transpose or batching rule of its own: those
    # live on the primitives.
    def _analyze(
        self, topology: SparseTopology, options: tuple, layout: Layout, precision: str
    ) -> Any:
        """Backend hook: the private symbolic analysis of one pattern.

        Called once per distinct content by
        [analyze][diff_soup.linear.analyze], which stores the result on the
        [Analysis][diff_soup.linear.Analysis].

        Args:
            topology: The sparsity pattern.
            options: The frozen backend options.
            layout: The validated layout.
            precision: The validated precision rung.

        Returns:
            Any backend-private object (``None`` when no symbolic work is needed).

        Raises:
            ConfigurationError: For an option or layout the backend does not support.
            NotImplementedError: On a backend that does not implement the contract.
        """
        raise NotImplementedError(_missing_hook_message(type(self), "_analyze"))

    def factor_avals(
        self, analysis: Analysis, values_aval: Any, *, orientation: str, precision: str, tol: Any
    ) -> tuple:
        """Backend hook: the abstract arrays of the factor handle for unbatched values.

        Args:
            analysis: The analysis.
            values_aval: The abstract ``(nnz,)`` values.
            orientation: ``"A"`` or ``"AT"``.
            precision: The precision rung.
            tol: ``None`` or a [Tolerance][diff_soup.linear.Tolerance].

        Returns:
            One ``jax.core.ShapedArray`` per handle array; the primitive layer
            prepends batch axes itself.

        Raises:
            NotImplementedError: On a backend that does not implement the contract.
        """
        raise NotImplementedError(_missing_hook_message(type(self), "factor_avals"))

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
        """Backend hook: factor ``values`` (JAX-level implementation).

        Accepts one optional leading batch axis on ``values``.

        Args:
            values: The operator values, ``(nnz,)`` or ``(B, nnz)``.
            token: The ``int32`` slot token, which tells the lanes of one
                call site apart.
            context_id: The ``int32`` scalar ``context_id`` of the owning
                [SolveContext][diff_soup.linear.SolveContext], or ``-1`` for
                a factorization without a context. A backend with native
                storage puts it in the handle leaves it returns, so solves
                and refactorizations find the context per lane.
            order: The factor's ``int32`` order word, one per lane.
            analysis: The analysis.
            orientation: ``"A"`` or ``"AT"``.
            precision: The precision rung.
            tol: ``None`` or a [Tolerance][diff_soup.linear.Tolerance].

        Returns:
            ``(*handle, status, order)`` with an ``int32`` status per lane and
            the order word: ``order`` itself, or the order word of the last
            primitive of the contract the hook binds.

        Raises:
            NotImplementedError: On a backend that does not implement the contract.
        """
        raise NotImplementedError(_missing_hook_message(type(self), "factor_fun"))

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
    ) -> tuple:
        """Backend hook: solve with a factor handle (JAX-level implementation).

        Accepts one optional leading batch axis on ``values``, the handle
        arrays, ``order`` and ``b``.

        An exact backend is called with exactly the arguments below and
        returns ``(x, status, order)``. An inexact backend (``is_exact =
        False``) is also called with the keywords ``weights`` and ``x0``, each
        ``None`` or an array of ``b``'s shape (the error weights, in the real
        dtype of the result, and the initial guess, in the result dtype),
        receives the resolved [Tolerance][diff_soup.linear.Tolerance] (never
        ``None``) as ``tol``, and returns ``(x, status, iterations,
        applications, order)`` with ``int32`` counts per lane. It stops when
        ``||w * (b - M x)||_wrms <= atol + rtol * ||w * b||_wrms`` (the
        weighted root-mean-square norm of ``Tolerance``, per right-hand
        side) and reports a status other than ``SUCCESS`` when it does not
        get there; its solution then stays finite.

        Args:
            values: The operator values the handle was built from.
            *handle: The factor handle arrays.
            order: The factor's ``int32`` order word, one per lane.
            b: Right-hand side, ``(n,)`` or ``(nrhs, n)``, already in the
                solve's result dtype: the precision rung's dtype, complex of
                the same width when the values or the right-hand side are
                complex.
            analysis: The analysis.
            orientation: ``"A"`` or ``"AT"``.
            transpose: Whether to solve with the transpose of the factored operator.
            precision: The precision rung.
            tol: ``None`` or a [Tolerance][diff_soup.linear.Tolerance].

        Returns:
            ``(x, status, order)`` with ``x`` shaped like ``b`` and in the
            result dtype, which is ``b``'s, and the order word (``order``
            itself, or the order word of the last primitive of the contract
            the hook binds); an inexact backend puts the iteration and
            application counts before the order word.

        Raises:
            NotImplementedError: On a backend that does not implement the contract.
        """
        raise NotImplementedError(_missing_hook_message(type(self), "apply_fun"))

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
    ) -> tuple:
        """Backend hook: refactor where ``predicate`` holds (JAX-level implementation).

        Accepts one optional leading batch axis on ``values``,
        ``held_values``, ``predicate``, the handle arrays and ``order``.
        Lanes whose predicate is false keep their handle unchanged and report
        the held factor's status (``SolveStatus.EMPTY`` for an empty handle).
        A backend with native storage reads the owning context from the
        handle leaves.

        Args:
            values: The new operator values.
            held_values: The values the held factor was built from, in the
                shape and dtype of ``values``.
            predicate: Boolean, one per lane.
            *handle: The held factor handle arrays.
            order: The factor's ``int32`` order word, one per lane.
            analysis: The analysis.
            orientation: ``"A"`` or ``"AT"``.
            precision: The precision rung.
            tol: ``None`` or a [Tolerance][diff_soup.linear.Tolerance].
            adopt: Refactor in place in the storage the held handle names,
                whichever call site built it, instead of storage of this
                call site. A backend whose factors are arrays, or that
                always refactors in place, accepts the flag without effect.

        Returns:
            ``(kept_values, *handle, status, order)``: ``kept_values`` is
            ``values`` in the lanes whose predicate holds and ``held_values``
            elsewhere; an ``int32`` status per lane; the order word (``order``
            itself, or the order word of the last primitive of the contract
            the hook binds).

        Raises:
            NotImplementedError: On a backend that does not implement the contract.
        """
        raise NotImplementedError(_missing_hook_message(type(self), "refactor_fun"))

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
        """Backend hook: the handle of a factor that holds no factorization yet.

        Called by [empty_factor][diff_soup.linear.empty_factor]. The leaves
        have exactly the shapes and dtypes ``factor_avals`` gives for
        ``values_aval`` and carry the backend's empty marker, which every
        solve, refactor and invalidation recognises.

        Args:
            analysis: The analysis.
            values_aval: The abstract ``(nnz,)`` values of the empty factor.
            context_id: The ``context_id`` of the owning
                [SolveContext][diff_soup.linear.SolveContext], ``-1``
                without one.
            orientation: ``"A"`` or ``"AT"``.
            precision: The precision rung.
            tol: ``None`` or a [Tolerance][diff_soup.linear.Tolerance].

        Returns:
            One concrete array per handle leaf.

        Raises:
            NotImplementedError: On a backend that does not implement the contract.
        """
        raise NotImplementedError(_missing_hook_message(type(self), "empty_handle"))

    def _release(self, ctx: SolveContext) -> None:
        """Backend hook: free every slot held for ``ctx`` (array-held factors: nothing).

        Called once by [SolveContext.release][diff_soup.linear.SolveContext.release]
        for each backend that factored in the context. A backend with native
        storage also records ``ctx.context_id`` as released, so a compiled
        program that later meets it fails at run time.
        """
        return None

    def _resident_bytes(self, ctx: SolveContext) -> int:
        """Backend hook: bytes of factor storage held for ``ctx`` (array-held factors: ``0``)."""
        return 0


def _missing_hook_message(cls: type, hook: str) -> str:
    """The error text for a backend that lacks an analyze/factor/solve contract hook."""
    return (
        f"{cls.__name__} does not implement the analyze/factor/solve contract: it defines no "
        f"{hook} hook. Implement _analyze, factor_avals, empty_handle and factor_fun, "
        "apply_fun and refactor_fun (or the lower_factor / lower_apply / lower_refactor "
        "lowerings), or use a backend that does, for example diff_soup.linear.Dense."
    )


# --------------------------------------------------------------------------- #
# The analyze / factor / solve API, with refactor and solve_checked
# --------------------------------------------------------------------------- #
_ANALYSES: dict[tuple, Analysis] = {}
_ANALYSES_LOCK = threading.Lock()


def _clear_analyses() -> None:
    """Drop every memoised [Analysis][diff_soup.linear.Analysis]; a test hook only.

    No library code path calls it. The one-equation programs of eager calls
    are dropped with the memo, and an analysis made again with equal content
    gets a native analysis of its own. A jitted program compiled against a
    dropped analysis is not dropped: it still names the dropped native
    analysis, whose storage is freed once the dropped ``Analysis`` is
    collected. Call it only where no such program runs afterwards.
    """
    with _ANALYSES_LOCK:
        _ANALYSES.clear()
    _compiled.cache_clear()


def _frozen_analysis_options(options: Any) -> tuple:
    """Normalise ``options`` (``None``, a dict, a dataclass or ``(key, value)`` pairs)."""
    try:
        frozen = _freeze_options(_options_dict(options))
        hash(frozen)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            "analyze() options must be None, a dict, a dataclass or (key, value) pairs with "
            f"hashable values; got {options!r} ({exc})."
        ) from exc
    return frozen


def analyze(
    topology: SparseTopology,
    *,
    backend: LinearSolver,
    options: Any = None,
    layout: Optional[Layout] = None,
    precision: str = "fp64",
    symmetric: bool = False,
    spd: bool = False,
) -> Analysis:
    """Analyze a sparsity pattern for one backend, before any values are factored.

    Equal arguments return the same `Analysis`, so the work runs once per pattern. A request
    the backend cannot serve raises here, before anything is traced.

    Args:
        topology: The `SparseTopology` of the matrix.
        backend: The backend object, for example `Dense()` or `KLU(ordering="colamd")`.
        options: Backend settings as a dict or a `CuDSSOptions`, overriding the fields of
            `backend` key by key.
        layout: The device placement, or `None` for one device.
        precision: The precision of the factors, `"fp64"` or `"fp32"`.
        symmetric: Declare the matrix symmetric, which is never detected from the values.
        spd: Declare the matrix symmetric positive definite, which implies `symmetric`.

    Returns:
        The `Analysis` to pass to `factor`.

    Raises:
        ConfigurationError: For a setting, precision or layout the backend does not support, or
            `symmetric` with Dirichlet rows.
    """
    if not isinstance(topology, SparseTopology):
        raise ConfigurationError(
            f"analyze() needs a diff_soup.linear.SparseTopology; got {type(topology).__name__}."
        )
    if not isinstance(backend, LinearSolver):
        raise ConfigurationError(
            "analyze() needs backend= a configured diff_soup.linear.LinearSolver, for example "
            f"diff_soup.linear.Dense(); got {type(backend).__name__}."
        )
    cls = type(backend)
    if cls._analyze is LinearSolver._analyze:
        raise ConfigurationError(_missing_hook_message(cls, "_analyze"))
    try:
        hash(backend)
    except TypeError as exc:
        raise ConfigurationError(
            f"analyze() needs a backend that is hashable by content; this {cls.__name__} is "
            f"not ({exc}). A backend used with analyze must be hashable by content and must "
            "not hold arrays."
        ) from exc
    if backend.factor_residency not in FACTOR_RESIDENCIES:
        raise ConfigurationError(
            f"{cls.__name__}.factor_residency must be one of {FACTOR_RESIDENCIES}; "
            f"got {backend.factor_residency!r}."
        )
    if backend.vmap_factors not in VMAP_FACTOR_MODES:
        raise ConfigurationError(
            f"{cls.__name__}.vmap_factors must be one of {VMAP_FACTOR_MODES}; "
            f"got {backend.vmap_factors!r}."
        )
    try:
        check_tolerance(getattr(backend, "default_tolerance", None))
    except FactorContractError as exc:
        raise ConfigurationError(
            f"{cls.__name__}.default_tolerance must be None or a valid "
            f"diff_soup.linear.Tolerance: {exc}"
        ) from exc
    if not isinstance(precision, str) or precision not in PRECISION_RUNGS:
        raise ConfigurationError(
            f"analyze() precision must be one of {PRECISION_RUNGS}; got {precision!r}."
        )
    if precision == "fp64" and not jax.config.read("jax_enable_x64"):
        raise ConfigurationError(
            "precision='fp64' needs JAX's 64-bit mode. Call "
            'jax.config.update("jax_enable_x64", True) before building arrays, export '
            "JAX_ENABLE_X64=1, or analyze at precision='fp32'."
        )
    if layout is None:
        layout = Layout()
    elif not isinstance(layout, Layout):
        raise ConfigurationError(
            f"analyze() layout must be None or a diff_soup.Layout; got {type(layout).__name__}."
        )
    backend._validate_capabilities(layout, precision)
    frozen = _frozen_analysis_options(options)
    carried = carried_options(backend)
    if carried:
        frozen = _freeze_options({**carried, **dict(frozen)})
    spd = bool(spd)
    symmetric = bool(symmetric) or spd
    if symmetric and topology.bc_indices is not None and int(topology.bc_indices.size) > 0:
        raise ConfigurationError(
            f"analyze() received {'spd' if spd else 'symmetric'}=True for a topology with "
            "Dirichlet rows, whose identity rows make the assembled operator unsymmetric; drop "
            "the declaration, or apply the boundary rows in the values and pass a topology "
            "without bc_indices."
        )
    key = (topology, backend, frozen, layout, precision, symmetric, spd)
    with _ANALYSES_LOCK:
        memoised = _ANALYSES.get(key)
    if memoised is not None:
        return memoised
    _check_dirichlet_diagonal(topology)
    built = Analysis(
        topology,
        backend,
        frozen,
        layout,
        precision,
        symmetric,
        spd,
        backend_analysis=backend._analyze(topology, frozen, layout, precision),
    )
    with _ANALYSES_LOCK:
        return _ANALYSES.setdefault(key, built)


def _check_dirichlet_diagonal(topology: SparseTopology) -> None:
    """Raise unless the pattern holds the diagonal entry of every Dirichlet row.

    A Dirichlet row is assembled as a unit row, which needs its diagonal
    position in the pattern; checking here keeps every backend identical.
    """
    bc = topology.bc_indices
    if bc is None or int(bc.size) == 0:
        return
    _, diagonal = dirichlet_rows(topology)
    missing = np.setdiff1d(np.unique(bc), diagonal)
    if missing.size:
        shown = ", ".join(str(int(row)) for row in missing[:10])
        more = f" and {missing.size - 10} more" if missing.size > 10 else ""
        raise ConfigurationError(
            f"the topology's Dirichlet rows [{shown}]{more} have no diagonal entry in the "
            "pattern; the topology must include the diagonal entry (i, i) of every Dirichlet "
            "row i."
        )


def _check_analysis(analysis: Any, caller: str) -> Analysis:
    """Raise unless ``analysis`` came from [analyze][diff_soup.linear.analyze]."""
    if not isinstance(analysis, Analysis):
        raise FactorContractError(
            f"{caller}() needs a diff_soup.linear.Analysis built by diff_soup.linear.analyze(); "
            f"got {type(analysis).__name__}."
        )
    return analysis


def _check_factor(f: Any, caller: str) -> Factor:
    """Raise unless ``f`` is a [Factor][diff_soup.linear.Factor]."""
    if not isinstance(f, Factor):
        raise FactorContractError(
            f"{caller}() needs a diff_soup.linear.Factor built by diff_soup.linear.factor(); "
            f"got {type(f).__name__}."
        )
    return f


def _check_values(analysis: Analysis, values: Any, caller: str) -> Array:
    """The operator values as an array of shape ``(nnz,)`` with a floating dtype."""
    values = jnp.asarray(values)
    if values.shape != (analysis.nnz,):
        raise FactorContractError(
            f"{caller}() needs values of shape ({analysis.nnz},), one per pattern entry of the "
            f"topology; got shape {tuple(values.shape)}. Batch independent operators with "
            "jax.vmap."
        )
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        raise FactorContractError(
            f"{caller}() needs floating or complex values; got dtype {values.dtype}."
        )
    return values


def _check_rhs(f: Factor, b: Any, caller: str) -> Array:
    """The right-hand side as ``(n,)`` or ``(nrhs, n)``, cast to the solve's result dtype.

    The result dtype is the precision rung's (``float64`` for ``"fp64"``,
    ``float32`` for ``"fp32"``), complex of the same width when the values or
    ``b`` are complex. This cast is the one place a right-hand side changes
    dtype; derivatives with respect to ``b`` pass back through it.
    """
    b = jnp.asarray(b)
    n = f.analysis.n
    if b.ndim not in (1, 2) or b.shape[-1] != n:
        raise FactorContractError(
            f"{caller}() needs a right-hand side of shape ({n},) or (nrhs, {n}); got shape "
            f"{tuple(b.shape)}. Batch independent right-hand sides with jax.vmap or stack them "
            "as rows."
        )
    result = solve_dtype(f.precision, f.values.dtype, b.dtype)
    return b if b.dtype == result else b.astype(result)


def _check_transpose(transpose: Any) -> bool:
    """``transpose`` as a Python bool (it selects the program, so it cannot be traced)."""
    if not isinstance(transpose, (bool, np.bool_)):
        raise FactorContractError(
            f"transpose must be a Python bool; got {type(transpose).__name__}. Select the "
            "orientation outside traced code."
        )
    return bool(transpose)


def _int32_operand(value: Any) -> Any:
    """An ``int32`` scalar operand: a NumPy scalar for a Python or NumPy int, else an array.

    A NumPy scalar becomes a literal of a traced program and lowers as a
    constant, which a backend's lowering can read (cuDSS then finds the
    factor's storage without reading the operand back from the device). A
    traced value stays traced.
    """
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return np.int32(value)
    return jnp.asarray(value, jnp.int32)


def _register_context(analysis: Analysis, ctx: Any, caller: str) -> None:
    """Register ``analysis``'s backend with ``ctx``, or refuse a missing or foreign context.

    A context is required for a backend whose ``factor_residency`` is not
    ``"arrays"``; a backend whose factors are arrays ignores it.
    """
    backend = analysis.backend
    if ctx is not None:
        if not isinstance(ctx, SolveContext):
            raise FactorContractError(
                f"{caller}() ctx must be None or a diff_soup.linear.SolveContext; got "
                f"{type(ctx).__name__}."
            )
        ctx.register(backend)
    elif backend.factor_residency != "arrays":
        raise FactorContractError(
            f"{public_backend_name(backend)} keeps its factors in {backend.factor_residency} "
            "storage, which a SolveContext owns; pass ctx=diff_soup.linear.SolveContext() to "
            f"{caller}()."
        )


def _followed(after: Any, caller: str) -> tuple[Factor, ...]:
    """The factors of an ``after=`` argument: one factor or a sequence of factors."""
    if isinstance(after, Factor):
        return (after,)
    try:
        others = tuple(after)
    except TypeError:
        others = None
    if others is None or not all(isinstance(other, Factor) for other in others):
        raise FactorContractError(
            f"{caller}() takes after= as a diff_soup.linear.Factor or a sequence of factors whose "
            f"operations the new factor follows; got {after!r}."
        )
    return others


def _linked_order(order: Any, others: Sequence[Factor]) -> Any:
    """``order`` made a data dependency of every order word of ``others``.

    The maximum of ``order`` and each other order word (reduced over its
    lanes); the value means nothing, only the dependency does.
    """
    order = jnp.asarray(order, jnp.int32)
    for other in others:
        order = jnp.maximum(order, jnp.max(jnp.asarray(other.order, jnp.int32)))
    return order


def _with_order(f: Factor, order: Any) -> Factor:
    """``f`` with the order word ``order``."""
    return Factor(
        analysis=f.analysis,
        orientation=f.orientation,
        precision=f.precision,
        exact=f.exact,
        values=f.values,
        handle=f.handle,
        generation=f.generation,
        order=order,
        storage=f.storage,
    )


def after(f: Factor, *others: Factor) -> Factor:
    """Order the next operations on a factor after those on other factors.

    Use it for factors that share one named storage, so that work on one waits for the others.

    Args:
        f: The factor whose operations must wait.
        *others: The factors to wait for.

    Returns:
        `f`, ordered after `others`.
    """
    f = _check_factor(f, "after")
    others = tuple(_check_factor(other, "after") for other in others)
    if not others:
        return f
    return _with_order(f, _linked_order(f.order, others).astype(jnp.int32))


def factor(
    analysis: Analysis,
    values: ArrayLike,
    *,
    ctx: Optional[SolveContext] = None,
    orientation: str = "A",
    tol: Optional[Tolerance] = None,
    after: Any = (),
) -> Factor:
    """Factor a matrix from its values.

    Args:
        analysis: The `Analysis` of the pattern.
        values: One value per pattern entry, shape `(nnz,)`, with `jax.vmap` for a batch of
            matrices.
        ctx: The `SolveContext` that owns the factor, required by KLU, SuperLU and cuDSS.
        orientation: `"A"` factors the matrix and `"AT"` its transpose.
        tol: A `Tolerance` for a backend of your own, ignored by the built-in backends.
        after: A factor, or a sequence of factors, whose operations this one must follow.

    Returns:
        The `Factor`.

    Raises:
        FactorContractError: For values of the wrong shape, or no `ctx` where the backend needs
            one.
    """
    analysis = _check_analysis(analysis, "factor")
    orientation = check_orientation(orientation)
    tol = check_tolerance(tol)
    others = _followed(after, "factor")
    _register_context(analysis, ctx, "factor")
    values = _check_values(analysis, values, "factor")
    order = _linked_order(np.int32(0), others) if others else np.int32(0)
    *handle, _, order = phased_factor_p.bind(
        jax.lax.stop_gradient(values),
        _int32_operand(0),
        _int32_operand(-1 if ctx is None else ctx.context_id),
        order,
        analysis=analysis,
        orientation=orientation,
        precision=analysis.precision,
        tol=tol,
    )
    return Factor(
        analysis=analysis,
        orientation=orientation,
        precision=analysis.precision,
        exact=bool(analysis.backend.is_exact),
        values=values,
        handle=tuple(handle),
        generation=jnp.zeros((), jnp.int32),
        order=order,
    )


def empty_factor(
    analysis: Analysis,
    *,
    ctx: Optional[SolveContext] = None,
    orientation: str = "A",
    tol: Optional[Tolerance] = None,
    dtype: Any = None,
    storage: Optional[str] = None,
    after: Any = (),
) -> Factor:
    """A factor that holds no factorization yet, the start of a loop that refactors it.

    Its solves return `nan` with `SolveStatus.EMPTY` until a refactor with a true predicate
    fills it.

    Args:
        analysis: The `Analysis` of the pattern.
        ctx: The `SolveContext` that will own the factor, as for `factor`.
        orientation: `"A"` for the matrix and `"AT"` for its transpose.
        tol: As for `factor`.
        dtype: The dtype of the values, by default float64 or float32 to match the analysis.
        storage: A name of 1 to 64 characters, and the factors of one name in one context share
            one storage.
        after: A factor, or a sequence of factors, whose operations this one must follow.

    Returns:
        The empty `Factor`.

    !!! example

        ```python
        def step(f, v):
            f = refactor(f, v, True)
            x, f = solve(f, b)
            return f, x

        f, xs = jax.lax.scan(step, empty_factor(analysis), values)
        ```
    """
    analysis = _check_analysis(analysis, "empty_factor")
    orientation = check_orientation(orientation)
    tol = check_tolerance(tol)
    cls = type(analysis.backend)
    dtype = rung_dtype(analysis.precision) if dtype is None else np.dtype(dtype)
    if not jnp.issubdtype(dtype, jnp.inexact):
        raise FactorContractError(
            f"empty_factor() needs a floating or complex dtype for the factor's values; got "
            f"{dtype}."
        )
    if storage is not None and (not isinstance(storage, str) or not 0 < len(storage) <= 64):
        raise FactorContractError(
            "empty_factor() takes storage=None or a storage name of 1 to 64 characters; got "
            f"{storage!r}."
        )
    others = _followed(after, "empty_factor")
    hook = getattr(analysis.backend, "empty_handle", None)
    if hook is None:
        raise FactorContractError(_missing_hook_message(cls, "empty_handle"))
    _register_context(analysis, ctx, "empty_factor")
    values_aval = ShapedArray((analysis.nnz,), dtype)
    common = dict(orientation=orientation, precision=analysis.precision, tol=tol)
    try:
        leaves = hook(
            analysis,
            values_aval,
            context_id=-1 if ctx is None else int(ctx.context_id),
            **common,
        )
    except NotImplementedError as exc:
        raise FactorContractError(str(exc)) from exc
    handle = check_empty_handle(analysis, values_aval, leaves, **common)
    declare = getattr(analysis.backend, "declare_storage", None)
    if storage is not None and declare is not None and ctx is not None:
        declare(analysis, int(ctx.context_id), storage)
    return Factor(
        analysis=analysis,
        orientation=orientation,
        precision=analysis.precision,
        exact=bool(analysis.backend.is_exact),
        values=jnp.zeros((analysis.nnz,), dtype),
        handle=handle,
        generation=jnp.full((), -1, jnp.int32),
        order=_linked_order(jnp.zeros((), jnp.int32), others),
        storage=storage,
    )


def _check_extra(f: Factor, b: Array, value: Any, name: str, dtype: Any, caller: str) -> Any:
    """``weights`` or ``x0`` as an array of ``b``'s shape in ``dtype``; ``None`` stays ``None``.

    The array carries no derivative: a solve's derivatives flow through the
    factor's values and the right-hand side only.
    """
    if value is None:
        return None
    value = jnp.asarray(value)
    n = f.analysis.n
    if tuple(value.shape) not in ((n,), tuple(b.shape)):
        raise FactorContractError(
            f"{caller}() needs {name} of shape ({n},) or the right-hand side's shape "
            f"{tuple(b.shape)}; got shape {tuple(value.shape)}."
        )
    if not jnp.issubdtype(value.dtype, jnp.number) or (
        name == "weights" and jnp.issubdtype(value.dtype, jnp.complexfloating)
    ):
        kind = "real numbers" if name == "weights" else "numbers"
        raise FactorContractError(f"{caller}() needs {name} of {kind}; got dtype {value.dtype}.")
    value = jax.lax.stop_gradient(value)
    if value.dtype != dtype:
        value = value.astype(dtype)
    return jnp.broadcast_to(value, b.shape)


def _resolve_tolerance(f: Factor, tol: Any, caller: str) -> Optional[Tolerance]:
    """The stopping rule a solve of ``f`` binds: ``None`` for an exact factor.

    An inexact factor takes ``tol``, else its backend's ``default_tolerance``,
    and refuses a solve that has neither.
    """
    tol = check_tolerance(tol)
    if f.exact:
        return None
    if tol is None:
        tol = check_tolerance(getattr(f.analysis.backend, "default_tolerance", None))
    if tol is None:
        name = type(f.analysis.backend).__name__
        raise FactorContractError(
            f"{caller}() on a factor of the inexact backend {name} needs a stopping rule, and "
            f"{name}.default_tolerance is None: pass tol=diff_soup.linear.Tolerance(rtol, atol, "
            "max_iter) to the solve, or give the backend a default_tolerance."
        )
    return tol


def _bind_solve(
    f: Factor,
    b: Any,
    *,
    transpose: Any,
    tol: Any,
    weights: Any,
    x0: Any,
    caller: str,
) -> tuple:
    """Bind the solve primitive.

    Returns ``(x, status, counts, b, weights, f)``: ``counts`` is ``None``
    for an exact factor and the ``(iterations, applications)`` of an inexact
    one; ``b`` and ``weights`` are as bound (cast, ``weights`` broadcast to
    ``b``'s shape or ``None``); ``f`` is the factor with the order word the
    solve returned. On an exact factor the tolerance and the initial guess
    are dropped and the bind is the plain solve, whatever the caller passed,
    so the program is the plain solve's.
    """
    f = _check_factor(f, caller)
    transpose = _check_transpose(transpose)
    tol = _resolve_tolerance(f, tol, caller)
    b = _check_rhs(f, b, caller)
    weights = _check_extra(f, b, weights, "weights", rung_dtype(f.precision), caller)
    if f.exact:
        x, status, order = phased_solve_p.bind(
            f.values,
            *f.handle,
            f.order,
            b,
            n_handle=len(f.handle),
            analysis=f.analysis,
            orientation=f.orientation,
            transpose=transpose,
            precision=f.precision,
            tol=None,
            storage=f.storage,
        )
        return x, status, None, b, weights, _with_order(f, order)
    x0 = _check_extra(f, b, x0, "x0", b.dtype, caller)
    extras = tuple(extra for extra in (weights, x0) if extra is not None)
    x, status, iterations, applications, order = phased_solve_p.bind(
        f.values,
        *f.handle,
        f.order,
        b,
        *extras,
        n_handle=len(f.handle),
        analysis=f.analysis,
        orientation=f.orientation,
        transpose=transpose,
        precision=f.precision,
        tol=tol,
        storage=f.storage,
        has_weights=weights is not None,
        has_x0=x0 is not None,
    )
    return x, status, (iterations, applications), b, weights, _with_order(f, order)


def solve(
    f: Factor,
    b: ArrayLike,
    *,
    transpose: bool = False,
    tol: Optional[Tolerance] = None,
    weights: Optional[ArrayLike] = None,
    x0: Optional[ArrayLike] = None,
) -> tuple[Array, Factor]:
    """Solve `M x = b` with a factor of `M`, or `M^T x = b` with `transpose=True`.

    A factor that is singular, non-finite or empty solves to `nan`. An iterative solve that
    stops short of `tol` returns its last iterate, and `solve_checked` reports the status.

    Args:
        f: The factor of `M`.
        b: The right-hand side, shape `(n,)`, or `(nrhs, n)` for several.
        transpose: A Python bool, `True` to solve with the transpose.
        tol: The `Tolerance` of an iterative solve, by default the backend's own.
        weights: The error weights of an iterative solve's root-mean-square norm.
        x0: The initial guess of an iterative solve, zeros by default.

    Returns:
        The solution `x` and the factor to use next.
    """
    x, *_, f = _bind_solve(
        f, b, transpose=transpose, tol=tol, weights=weights, x0=x0, caller="solve"
    )
    return x, f


def refactor(f: Factor, values: ArrayLike, predicate: ArrayLike, *, adopt: bool = False) -> Factor:
    """Factor new values where `predicate` is true, and keep the factor where it is false.

    Under `jax.vmap` each matrix of the batch takes its own predicate.

    Args:
        f: The factor to update.
        values: The new values, in the dtype of `f.values`.
        predicate: A boolean scalar, `True` to refactor.
        adopt: `True` refactors in the storage of `f`, and `False` in storage of its own.

    Returns:
        The updated `Factor`.

    Raises:
        FactorContractError: For values of another shape or dtype, or a predicate that is not a
            scalar.
    """
    f = _check_factor(f, "refactor")
    values = _check_values(f.analysis, values, "refactor")
    if values.dtype != f.values.dtype:
        raise FactorContractError(
            f"refactor() needs values of the factor's dtype {f.values.dtype}; got "
            f"{values.dtype}. Cast the values before refactoring."
        )
    predicate = jnp.asarray(predicate)
    if predicate.shape != ():
        raise FactorContractError(
            "refactor() takes one boolean predicate per lane (shape ()); got shape "
            f"{tuple(predicate.shape)}. Batch lanes with jax.vmap."
        )
    predicate = jax.lax.stop_gradient(predicate).astype(jnp.bool_)
    if adopt is not True and adopt is not False:
        raise FactorContractError(
            f"refactor() takes adopt=True or adopt=False; got {adopt!r}. The flag is static: it "
            "selects the storage the refactorization runs in and cannot be traced."
        )
    # Both value operands go to the backend, which returns the values the
    # result carries (the new ones where the predicate holds, the held ones
    # elsewhere); a backend that refactors a whole batch when any lane asks
    # (cuDSS) refactors a lane that did not ask from the values it holds.
    kept, *handle, _, order = phased_refactor_p.bind(
        values,
        f.values,
        predicate,
        *f.handle,
        f.order,
        n_handle=len(f.handle),
        analysis=f.analysis,
        orientation=f.orientation,
        precision=f.precision,
        tol=None,
        adopt=adopt,
        storage=f.storage,
    )
    generation = jnp.where(predicate, f.generation + 1, f.generation).astype(jnp.int32)
    return Factor(
        analysis=f.analysis,
        orientation=f.orientation,
        precision=f.precision,
        exact=f.exact,
        values=kept,
        handle=tuple(handle),
        generation=generation,
        order=order,
        storage=f.storage,
    )


def invalidate(factor: Factor) -> Factor:
    """Mark a factor's stored factorization out of date, so its next solve refactors from its values.

    Use it when the storage may hold another factorization than the factor's values describe.
    A factor held in arrays, as on `Dense`, has nothing to mark.

    Args:
        factor: The factor.

    Returns:
        The factor to use next.
    """
    factor = _check_factor(factor, "invalidate")
    (order,) = phased_invalidate_p.bind(
        *factor.handle,
        _int32_operand(-1),
        factor.order,
        n_handle=len(factor.handle),
        analysis=factor.analysis,
        orientation=factor.orientation,
        precision=factor.precision,
        storage=factor.storage,
    )
    return _with_order(factor, order)


def _residual_of(f: Factor, b: Array, x: Array, transpose: bool) -> Array:
    """``b - M x`` for the operator a solve applied, at the precision rung (not differentiated).

    ``b`` is the cast right-hand side and ``x`` the solution, both in the
    solve's result dtype; the values are cast to it too.
    """
    values = jax.lax.stop_gradient(f.values).astype(x.dtype)
    x = jax.lax.stop_gradient(x)
    b = jax.lax.stop_gradient(b)
    applied = apply_assembled(
        f.analysis, values, x, transpose=effective_transpose(f.orientation, transpose)
    )
    return b - applied


def _max_residual(f: Factor, b: Array, x: Array, transpose: bool) -> Array:
    """``max|b - M x|`` for the operator a solve applied (not differentiated)."""
    residual = jnp.abs(_residual_of(f, b, x, transpose))
    if residual.size == 0:
        return jnp.zeros((), residual.dtype)
    return jnp.max(residual)


def _weighted_residual(f: Factor, b: Array, x: Array, weights: Array, transpose: bool) -> Array:
    """``||w * (b - M x)||_wrms``, the largest over a block's right-hand sides (not differentiated).

    ``weights`` has ``b``'s shape; the norm is
    ``sqrt(mean_i |w_i r_i|^2)`` over the last axis.
    """
    weighted = jnp.abs(jax.lax.stop_gradient(weights) * _residual_of(f, b, x, transpose))
    if weighted.size == 0:
        return jnp.zeros((), weighted.dtype)
    return jnp.max(jnp.sqrt(jnp.mean(weighted**2, axis=-1)))


def solve_checked(
    f: Factor,
    b: ArrayLike,
    *,
    transpose: bool = False,
    tol: Optional[Tolerance] = None,
    weights: Optional[ArrayLike] = None,
    x0: Optional[ArrayLike] = None,
) -> tuple[Array, Factor, SolveReport]:
    """Solve as `solve` does, and also return a `SolveReport` with the status and the residual.

    Args:
        f: The factor of `M`.
        b: The right-hand side, shape `(n,)`, or `(nrhs, n)` for several.
        transpose: A Python bool, `True` to solve with the transpose.
        tol: The `Tolerance` of an iterative solve, by default the backend's own.
        weights: The error weights, which also make the reported residual a weighted
            root-mean-square norm.
        x0: The initial guess of an iterative solve, zeros by default.

    Returns:
        The solution `x`, the factor to use next, and the `SolveReport`.
    """
    x, status, counts, b, weights, f = _bind_solve(
        f, b, transpose=transpose, tol=tol, weights=weights, x0=x0, caller="solve_checked"
    )
    if weights is None:
        residual = _max_residual(f, b, x, bool(transpose))
    else:
        residual = _weighted_residual(f, b, x, weights, bool(transpose))
    if counts is None:
        iterations, applications = jnp.zeros((), jnp.int32), jnp.ones((), jnp.int32)
    else:
        iterations, applications = (jnp.asarray(count, jnp.int32) for count in counts)
    report = SolveReport(
        status=jnp.asarray(status, jnp.int32),
        residual=residual,
        iterations=iterations,
        applications=applications,
    )
    return x, f, report
