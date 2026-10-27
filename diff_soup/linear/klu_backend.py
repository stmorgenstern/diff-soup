"""SuiteSparse KLU sparse-direct backend (host CPU) on the analyze / factor / solve contract.

[KLUBackend][diff_soup.linear.klu_backend.KLUBackend], exported as
``diff_soup.linear.KLU``, factors real float64 matrices with SuiteSparse
**KLU** through the native ``diff_soup._internal.klu_ffi`` extension
(``_native_klu``). The symbolic analysis of a pattern is built once on the
host; numeric factors live in host memory owned by a
[SolveContext][diff_soup.linear.SolveContext], and every native call takes
the factor's order word and returns it, so the operations on a factor that
is passed from one to the next keep that order under ``jit``, ``vmap`` and
both autodiff modes.

Dtype contract: KLU factorizes real double-precision matrices only. Complex
values or right-hand sides are refused at trace time with
[FactorContractError][diff_soup.linear.FactorContractError] (a complex system
must be embedded as its real ``2n`` block form, or solved with a backend that
factors complex values).

Telemetry: [native_stats][diff_soup.linear.klu_backend.KLUBackend.native_stats]
reports the process-wide counters of the extension (factorizations,
refactorizations, rebuilds, solves, failures and live storage), counted when
the native call runs.
"""

from __future__ import annotations

import weakref
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup._internal import klu_ffi
from diff_soup._layout import Layout
from diff_soup.linear._api import LinearSolver, SparseTopology, option_field_items
from diff_soup.linear._factor import (
    Analysis,
    FactorContractError,
    SolveContext,
    dirichlet_rows,
)
from diff_soup.linear._primitive import (
    cast_operand,
    empty_slot_handle,
    label_empty_slots,
    lower_ffi_call,
    lower_slot_apply,
    lower_slot_refactor,
    lowered_site_key,
    storage_key,
)


def klu_available() -> bool:
    """Return True if the native KLU extension imports and registers its FFI targets."""
    try:
        return klu_ffi.available()
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Factor contract: the native extension diff_soup._internal.klu_ffi
# --------------------------------------------------------------------------- #
#: Fill-reducing orderings KLU applies to each diagonal block, by option value.
KLU_ORDERINGS: dict[str, int] = {"amd": 0, "colamd": 1}

#: Row scalings KLU applies before pivoting, by option value.
KLU_SCALINGS: dict[str, int] = {"none": 0, "sum": 1, "max": 2}

_KLU_OPTION_NAMES = ("ordering", "btf", "scale", "pivot_tolerance")

_FACTOR_TARGET = "diffsoup_klu_factor_f64"
_REFACTOR_TARGET = "diffsoup_klu_refactor_f64"
_APPLY_TARGET = "diffsoup_klu_apply_f64"


def _native():
    """The loaded KLU extension (registers its FFI targets on first use).

    Raises:
        ConfigurationError: If the extension is not built in this checkout,
            with the commands that build it.
    """
    try:
        return klu_ffi.load()
    except ImportError as exc:
        raise ConfigurationError(str(exc)) from exc


class _KLUContractAnalysis:
    """Host record of one analysed pattern: the native analysis id and sizes."""

    __slots__ = ("native_id", "n", "nnz", "ordering", "__weakref__")

    def __init__(self, native_id: int, n: int, nnz: int, ordering: str):
        """Record a native analysis; it is released when this record is collected."""
        self.native_id = int(native_id)
        self.n = int(n)
        self.nnz = int(nnz)
        self.ordering = ordering
        weakref.finalize(self, _release_native_analysis, self.native_id)

    def __repr__(self) -> str:
        return (
            f"_KLUContractAnalysis(id={self.native_id}, n={self.n}, nnz={self.nnz}, "
            f"ordering={self.ordering!r})"
        )


def _release_native_analysis(native_id: int) -> None:
    """Drop the native registry's reference to an analysis (its sites keep it alive)."""
    try:
        _native().analysis_release(native_id)
    except Exception:  # noqa: BLE001 - interpreter shutdown or a missing module
        pass


def _parse_contract_options(options: tuple) -> dict[str, Any]:
    """Validate the options [analyze][diff_soup.linear.analyze] passes to KLU."""
    opts = dict(options)
    unknown = sorted(set(opts) - set(_KLU_OPTION_NAMES))
    if unknown:
        raise ConfigurationError(
            f"KLU does not take the options {unknown}; it accepts {list(_KLU_OPTION_NAMES)}."
        )
    ordering = opts.get("ordering", "amd")
    if ordering not in KLU_ORDERINGS:
        raise ConfigurationError(
            f"KLU option ordering must be one of {sorted(KLU_ORDERINGS)}; got {ordering!r}."
        )
    btf = opts.get("btf", True)
    if not isinstance(btf, (bool, np.bool_)):
        raise ConfigurationError(f"KLU option btf must be True or False; got {btf!r}.")
    scale = opts.get("scale", "max")
    if scale not in KLU_SCALINGS:
        raise ConfigurationError(
            f"KLU option scale must be one of {sorted(KLU_SCALINGS)}; got {scale!r}."
        )
    tolerance = opts.get("pivot_tolerance", 1e-3)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float, np.floating)):
        raise ConfigurationError(
            f"KLU option pivot_tolerance must be a number in (0, 1]; got {tolerance!r}."
        )
    if not 0.0 < float(tolerance) <= 1.0:
        raise ConfigurationError(
            f"KLU option pivot_tolerance must lie in (0, 1] (1 is partial pivoting); got "
            f"{tolerance!r}."
        )
    return {
        "ordering": ordering,
        "btf": bool(btf),
        "scale": scale,
        "pivot_tolerance": float(tolerance),
    }


def _site_key(ctx: Any, kind: str, analysis: Analysis, *parts: Any) -> np.int64:
    """The native site key of one lowered factor or refactor call on ``analysis``.

    A deterministic key of the lowered call
    ([lowered_site_key][diff_soup.linear._primitive.lowered_site_key]): the
    handler creates the site on the first call that carries it, so two
    lowerings of one program share their sites and their program text. A
    site belongs to no context: its slots are keyed by the context id the
    call carries at run time, and each slot is freed with its context.
    ``parts`` are the call's parameters and operand shapes, which tell
    calls of otherwise equal programs apart.
    """
    record: _KLUContractAnalysis = analysis.backend_analysis
    shapes = tuple((tuple(aval.shape), str(aval.dtype)) for aval in ctx.avals_in)
    return lowered_site_key(ctx, f"klu {kind}", record.native_id, shapes, *parts)


def _analysis_id(analysis: Analysis) -> np.int64:
    """The native id of ``analysis``: solve and refactor calls refuse a slot built for another."""
    record: _KLUContractAnalysis = analysis.backend_analysis
    return np.int64(record.native_id)


def _check_real(dtype: Any, what: str = "values") -> None:
    """Raise unless ``dtype`` is a real floating dtype."""
    if not jnp.issubdtype(dtype, jnp.floating):
        raise FactorContractError(
            f"KLU factors real matrices only; got {what} of dtype {jnp.dtype(dtype)}. Embed a "
            "complex system in its real 2n block form [[Re, -Im], [Im, Re]] or use a backend "
            "that factors complex values."
        )


class KLUBackend(LinearSolver):
    """SuiteSparse KLU, a sparse direct solver on the CPU.

    It factors float64 matrices only. `linear="klu"` selects it with default settings, and
    `"auto"` picks it on a CPU host.

    Args:
        ordering: The fill-reducing ordering of each block, `"amd"` (default) or `"colamd"`.
        btf: `True` (default) permutes to block triangular form first and factors each diagonal
            block alone.
        scale: The row scaling before pivoting, `"max"` (default), `"sum"` or `"none"`.
        pivot_tolerance: The diagonal pivoting threshold in `(0, 1]`, where `0.001` (default)
            favours the diagonal and `1.0` is partial pivoting.

    Raises:
        ConfigurationError: For a setting outside these values.
    """

    ordering: str = eqx.field(static=True, default="amd")
    btf: bool = eqx.field(static=True, default=True)
    scale: str = eqx.field(static=True, default="max")
    pivot_tolerance: float = eqx.field(static=True, default=1e-3)

    backend_name = "klu"
    option_fields = _KLU_OPTION_NAMES
    platforms = ("cpu",)

    factor_residency = "host"
    transpose_is_free = True
    vmap_factors = "sequential"
    supported_precisions = ("fp64",)
    is_exact = True
    accepts_operator = False

    def __check_init__(self) -> None:
        """Reject a setting ``analyze`` would refuse, when the backend is built."""
        _parse_contract_options(option_field_items(self))

    def _analyze(
        self, topology: SparseTopology, options: tuple, layout: Layout, precision: str
    ) -> _KLUContractAnalysis:
        """Build the compressed-column structure and KLU symbolic factorization of ``topology``.

        Args:
            topology: The sparsity pattern; its Dirichlet rows must have a
                diagonal entry in the pattern.
            options: Frozen backend options (see the class docstring).
            layout: The validated layout (single-process host only).
            precision: The validated precision rung (``"fp64"``).

        Returns:
            The host record of the native analysis.

        Raises:
            ConfigurationError: For an unknown or invalid option, a pattern
                KLU cannot analyse, or a missing native extension.
        """
        del layout, precision  # analyze() already rejected multi-device layouts and fp32
        parsed = _parse_contract_options(options)
        native = _native()
        n = int(topology.n_dofs)
        rows = np.ascontiguousarray(np.asarray(topology.rows), dtype=np.int32)
        cols = np.ascontiguousarray(np.asarray(topology.cols), dtype=np.int32)
        mask, diagonal = dirichlet_rows(topology)
        zeroed = np.zeros(rows.shape, dtype=np.bool_) if mask is None else np.asarray(mask)
        try:
            native_id = native.analysis_create(
                n,
                rows,
                cols,
                np.ascontiguousarray(zeroed, dtype=np.bool_),
                np.ascontiguousarray(diagonal, dtype=np.int32),
                KLU_ORDERINGS[parsed["ordering"]],
                parsed["btf"],
                KLU_SCALINGS[parsed["scale"]],
                parsed["pivot_tolerance"],
            )
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        return _KLUContractAnalysis(native_id, n, int(rows.size), parsed["ordering"])

    def factor_avals(
        self, analysis: Analysis, values_aval: Any, *, orientation: str, precision: str, tol: Any
    ) -> tuple:
        """The handle: two ``int32`` scalars, the owning context's id and the slot's id.

        Raises:
            FactorContractError: If the values are not real.
        """
        _check_real(values_aval.dtype)
        scalar = jax.core.ShapedArray((), jnp.int32)
        return (scalar, scalar)

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
        """The handle of an empty factor: the context's id and slot ``-1``.

        Raises:
            FactorContractError: If the values are not real.
        """
        _check_real(values_aval.dtype)
        return empty_slot_handle(context_id)

    def label_empty_lanes(self, handle: tuple, lanes: Any, size: int) -> tuple:
        """Give every empty lane of a lane-by-lane batch a slot marker of its own.

        A refactor under ``vmap`` maps over the lanes, so without this the
        empty lanes of a batch would all be factored into one slot of the
        call site; see
        [label_empty_slots][diff_soup.linear._primitive.label_empty_slots].
        """
        return label_empty_slots(handle, lanes, size)

    def lower_factor(
        self,
        ctx: Any,
        values: Any,
        token: Any,
        context_id: Any,
        order: Any,
        *,
        analysis: Analysis,
        orientation: str,
        precision: str,
        tol: Any,
    ) -> list:
        """Lower a factorization of ``A_bc`` to one FFI call ordered by the order word.

        Each lowered call names a native site on the analysis by a key
        computed from the call (created on its first run). The handler
        keeps one slot per ``(context id, slot token)`` at that site, creating
        it on first use and refactoring it in place on later runs, so the
        lanes of a value batch (distinct slot tokens) get one slot each and
        every context gets its own. Both orientations factor ``A_bc``; a
        solve with orientation ``"AT"`` runs the transposed KLU solve.

        Returns:
            The MLIR values of ``(context id, slot, status, order)``.
        """
        del tol
        site = _site_key(ctx, "factor", analysis, orientation, precision)
        values, values_aval = cast_operand(ctx, values, ctx.avals_in[0], np.float64)
        results, order = lower_ffi_call(
            ctx,
            _FACTOR_TARGET,
            order,
            [values, token, context_id],
            [values_aval, *ctx.avals_in[1:3]],
            ctx.avals_out[:-1],
            site=site,
            analysis_id=_analysis_id(analysis),
            storage=np.int64(0),
        )
        return [*results, order]

    def lower_apply(
        self,
        ctx: Any,
        values: Any,
        held_context: Any,
        slot: Any,
        order: Any,
        b: Any,
        *,
        n_handle: int,
        analysis: Analysis,
        orientation: str,
        transpose: bool,
        precision: str,
        tol: Any,
        storage: Optional[str] = None,
    ) -> list:
        """Lower a solve with the slot's factors to one FFI call ordered by the order word.

        Runs ``klu_tsolve`` when exactly one of ``orientation == "AT"`` and
        ``transpose`` holds, else ``klu_solve``. ``b`` of shape ``(nrhs, n)``
        is solved as one block. The solution is ``float64``, the result
        dtype of the ``"fp64"`` rung, which ``b`` already has; it is ``nan``
        unless the factorization succeeded. The call fails when the handle's
        context has been released or its slot was built for another analysis
        (the call carries the analysis's native id as ``analysis_id``). An
        empty handle (slot ``-1``) makes no call and returns ``nan`` with
        ``SolveStatus.EMPTY``.

        A solve with ``transpose=False`` of a factor that names shared
        storage (``storage``) uses the storage's slot as it stands when a
        refactor of that slot ran earlier in the same execution; any
        other solve first refactors a slot that holds other values than the
        ones it carries.

        Returns:
            The MLIR values of ``(x, status, order)``.
        """
        del n_handle, precision, tol
        return lower_slot_apply(
            ctx,
            values,
            held_context,
            slot,
            order,
            b,
            target=_APPLY_TARGET,
            values_dtype=np.float64,
            transpose=(orientation == "AT") != bool(transpose),
            analysis_id=_analysis_id(analysis),
            storage=storage_key(None if transpose else storage),
        )

    def lower_refactor(
        self,
        ctx: Any,
        values: Any,
        held_values: Any,
        predicate: Any,
        held_context: Any,
        held_slot: Any,
        order: Any,
        *,
        n_handle: int,
        analysis: Analysis,
        orientation: str,
        precision: str,
        tol: Any,
        adopt: bool,
        storage: Optional[str] = None,
    ) -> list:
        """Lower a refactorization where ``predicate`` holds to one ordered FFI call.

        Each lowered call names a native site by a key computed from the
        call (created on its first run). A held slot that is already
        one of the site's slots is refactored in place with ``klu_refactor``
        (a loop carrying the factor keeps one slot and its pivot sequence);
        any other held slot gets the site's slot for it in the handle's
        context, so other holders of that factor keep it. A false predicate
        returns the held handle and that slot's status. The call fails when
        the held slot was built for another analysis. An empty handle (slot
        ``-1``) whose predicate holds is factored into the site's own slot,
        and one whose predicate is false stays empty with
        ``SolveStatus.EMPTY``. The kept values are ``values`` where the
        predicate holds and ``held_values`` elsewhere. ``adopt`` is
        accepted without effect: this backend has one rule for a held slot
        of another site. A factor that names shared storage (``storage``)
        is refactored in place in that storage's slot, and with a false
        predicate only when the slot holds other values than the ones the
        factor keeps; an empty handle fills the storage's slot.

        Returns:
            The MLIR values of ``(kept values, context id, slot, status, order)``.
        """
        del n_handle, tol
        return lower_slot_refactor(
            ctx,
            values,
            held_values,
            predicate,
            held_context,
            held_slot,
            order,
            refactor_target=_REFACTOR_TARGET,
            factor_target=_FACTOR_TARGET,
            site=_site_key(ctx, "refactor", analysis, orientation, precision, adopt, storage),
            analysis_id=_analysis_id(analysis),
            values_dtype=np.float64,
            storage=storage_key(storage),
        )

    def _release(self, ctx: SolveContext) -> None:
        """Free every slot held for ``ctx`` and record its id as released."""
        _native().context_release(int(ctx.context_id))

    def _resident_bytes(self, ctx: SolveContext) -> int:
        """Bytes of the slots held for ``ctx``: value copies and the numeric factorizations."""
        return int(_native().resident_bytes(int(ctx.context_id)))

    @staticmethod
    def native_stats() -> dict:
        """Process-wide counters of the KLU extension.

        Returns:
            A dict with ``factorizations`` (``klu_factor`` runs),
            ``refactorizations`` (``klu_refactor`` results kept),
            ``refactor_fallbacks`` (``klu_refactor`` results replaced by a
            fresh ``klu_factor``), ``rebuilds`` (factorizations a solve started
            because its slot held other values), ``solves`` (``klu_solve`` and
            ``klu_tsolve`` calls), ``nonfinite`` and ``singular`` (factor
            requests with non-finite values or a singular result),
            ``released_context_errors`` (calls refused because their context
            was released), ``live_slots``, ``live_sites`` and
            ``live_analyses``.

        Raises:
            ConfigurationError: If the KLU extension is not built.
        """
        return dict(_native().stats())

    @staticmethod
    def reset_native_stats() -> None:
        """Zero the counters of [native_stats][diff_soup.linear.klu_backend.KLUBackend.native_stats].

        Raises:
            ConfigurationError: If the KLU extension is not built.
        """
        _native().reset_stats()
