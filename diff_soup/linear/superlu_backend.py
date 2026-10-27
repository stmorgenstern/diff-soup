"""SuperLU_MT backend: threaded sparse LU on the host.

[SuperLU][diff_soup.linear.superlu_backend.SuperLU] implements the
[analyze][diff_soup.linear.analyze] / [factor][diff_soup.linear.factor] /
[solve][diff_soup.linear.solve] contract with SuperLU_MT, a shared-memory
parallel sparse LU with partial pivoting (BSD-3; see
``diff_soup/_internal/superlu_ffi/THIRD_PARTY_NOTICES.md``).

- ``analyze`` builds the compressed-column structure of the pattern and its
  fill-reducing column ordering once per pattern, on the host.
- ``factor`` runs ``pdgstrf`` on every core the process may use and keeps
  the factors in host memory owned by a
  [SolveContext][diff_soup.linear.SolveContext]. The factor's handle is two
  ``int32`` scalars: the context's ``context_id`` and the slot that holds
  the factors. Each compiled ``factor`` call keeps one slot per context and
  slot token, so the lanes of a value batch each get their own slot,
  every context gets its own, and a compiled program reuses its slots on
  every run.
- ``refactor`` with the same pattern reuses the column ordering, the
  elimination tree and the storage of ``L`` and ``U`` (``pdgstrf`` with
  ``refact = YES``); a loop that carries a factor refactors one slot in place.
- ``solve`` runs ``dgstrs`` on all right-hand sides as one block, with
  ``trans`` selecting the orientation, so both ``M x = b`` and ``M^T x = b``
  use one factorization.

A solve whose values differ from the values its slot was last factored with
refactors the slot from the values it carries first, so it never reads a
factor built from other values. An empty factor
([empty_factor][diff_soup.linear.empty_factor]) names slot ``-1``: its
solves return ``nan`` with ``SolveStatus.EMPTY`` and call no library, and
the first refactor whose predicate holds factors it into a slot of that
refactor's call site.

Every native call takes the factor's order word as an operand and returns
it, so the operations on a factor that is passed from one to the next run in
that order: a refactorization of a slot never runs before a solve whose
returned factor it refactors. Releasing a context frees its slots; a call
that meets a released context fails with an error naming it.

The backend needs the native extension ``diff_soup._internal.superlu_ffi``;
analysis raises [ConfigurationError][diff_soup._errors.ConfigurationError]
with build instructions when it is missing. It factors real float64
operators only (``supported_precisions == ("fp64",)``).
"""

from __future__ import annotations

import weakref
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup._internal import superlu_ffi
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

__all__ = ["SuperLU", "superlu_available"]


def superlu_available() -> bool:
    """Return True if the native SuperLU_MT extension imports and registers its FFI targets."""
    try:
        return superlu_ffi.available()
    except Exception:  # noqa: BLE001
        return False


def require_superlu(requested_by: str) -> None:
    """Raise unless the native SuperLU_MT extension loads.

    Args:
        requested_by: What selected the backend, as the user wrote it (for
            example ``"linear='superlu'"``); the message starts with it.

    Raises:
        RuntimeError: When the extension is not built on this host, with the
            commands that build it.
    """
    try:
        superlu_ffi.load()
    except ImportError as exc:
        raise RuntimeError(
            f"{requested_by} selects the SuperLU backend, whose native SuperLU_MT extension is "
            f"not available on this host; select 'klu' or 'dense', or build it. "
            f"{str(exc).rstrip()}"
        ) from exc


#: Column orderings ``get_perm_c`` offers, by option value.
ORDERINGS: dict[str, int] = {
    "natural": 0,
    "mmd_ata": 1,
    "mmd_at_plus_a": 2,
    "colamd": 3,
}

_OPTION_NAMES = ("ordering", "panel_size", "relax", "pivot_threshold")

_FACTOR_TARGET = "diffsoup_superlu_factor_f64"
_REFACTOR_TARGET = "diffsoup_superlu_refactor_f64"
_APPLY_TARGET = "diffsoup_superlu_apply_f64"


def _native():
    """The loaded native module (registers the FFI targets on first use)."""
    return superlu_ffi.load()


class _SuperLUAnalysis:
    """Host record of one analysed pattern: the native analysis id and sizes."""

    __slots__ = ("native_id", "n", "nnz", "ordering", "__weakref__")

    def __init__(self, native_id: int, n: int, nnz: int, ordering: str):
        """Record a native analysis; it is released when this record is collected."""
        self.native_id = int(native_id)
        self.n = int(n)
        self.nnz = int(nnz)
        self.ordering = ordering
        weakref.finalize(self, _release_analysis, self.native_id)

    def __repr__(self) -> str:
        return (
            f"_SuperLUAnalysis(id={self.native_id}, n={self.n}, nnz={self.nnz}, "
            f"ordering={self.ordering!r})"
        )


def _release_analysis(native_id: int) -> None:
    """Drop the native registry's reference to an analysis (its slots keep it alive)."""
    try:
        _native().analysis_release(native_id)
    except Exception:  # noqa: BLE001 - interpreter shutdown or a missing module
        pass


def _parse_options(options: tuple) -> dict[str, Any]:
    """Validate the backend options of [SuperLU][diff_soup.linear.superlu_backend.SuperLU]."""
    opts = dict(options)
    unknown = sorted(set(opts) - set(_OPTION_NAMES))
    if unknown:
        raise ConfigurationError(
            f"SuperLU does not take the options {unknown}; it accepts {list(_OPTION_NAMES)}."
        )
    ordering = opts.get("ordering", "colamd")
    if ordering not in ORDERINGS:
        raise ConfigurationError(
            f"SuperLU option ordering must be one of {sorted(ORDERINGS)}; got {ordering!r}."
        )
    parsed: dict[str, Any] = {"ordering": ordering}
    for name in ("panel_size", "relax"):
        value = opts.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ConfigurationError(
                f"SuperLU option {name} must be a non-negative integer (0 selects the library "
                f"default); got {value!r}."
            )
        parsed[name] = int(value)
    threshold = opts.get("pivot_threshold", 1.0)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.floating)):
        raise ConfigurationError(
            f"SuperLU option pivot_threshold must be a number in [0, 1]; got {threshold!r}."
        )
    if not 0.0 <= float(threshold) <= 1.0:
        raise ConfigurationError(
            f"SuperLU option pivot_threshold must lie in [0, 1] (1 is partial pivoting); got "
            f"{threshold!r}."
        )
    parsed["pivot_threshold"] = float(threshold)
    return parsed


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
    record: _SuperLUAnalysis = analysis.backend_analysis
    shapes = tuple((tuple(aval.shape), str(aval.dtype)) for aval in ctx.avals_in)
    return lowered_site_key(ctx, f"superlu {kind}", record.native_id, shapes, *parts)


def _analysis_id(analysis: Analysis) -> np.int64:
    """The native id of ``analysis``: solve and refactor calls refuse a slot built for another."""
    record: _SuperLUAnalysis = analysis.backend_analysis
    return np.int64(record.native_id)


def _check_real(dtype: Any, what: str = "values") -> None:
    """Raise unless ``dtype`` is a real floating dtype."""
    if not jnp.issubdtype(dtype, jnp.floating):
        raise FactorContractError(
            f"SuperLU solves real systems; got {what} of dtype {jnp.dtype(dtype)}. Embed a "
            "complex system in its real 2n form or use a backend that factors complex values."
        )


class SuperLU(LinearSolver):
    """SuperLU_MT, a threaded sparse direct solver on the CPU.

    It factors float64 matrices only. `linear="superlu"` selects it with default settings, and
    `"auto"` never picks it. It needs a source build with SuperLU_MT.

    Args:
        ordering: The fill-reducing column ordering, `"colamd"` (default), `"mmd_ata"`,
            `"mmd_at_plus_a"` or `"natural"`.
        panel_size: The panel width, where `0` (default) keeps SuperLU_MT's own value.
        relax: The relaxed supernode size, where `0` (default) keeps SuperLU_MT's own value.
        pivot_threshold: The diagonal pivoting threshold in `[0, 1]`, where `1.0` (default) is
            partial pivoting.

    Raises:
        ConfigurationError: For a setting outside these values.
    """

    ordering: str = eqx.field(static=True, default="colamd")
    panel_size: int = eqx.field(static=True, default=0)
    relax: int = eqx.field(static=True, default=0)
    pivot_threshold: float = eqx.field(static=True, default=1.0)

    backend_name = "superlu"
    option_fields = _OPTION_NAMES
    platforms = ("cpu",)

    factor_residency = "host"
    transpose_is_free = True
    vmap_factors = "sequential"
    supported_precisions = ("fp64",)
    is_exact = True
    accepts_operator = False

    def __check_init__(self) -> None:
        """Reject a setting ``analyze`` would refuse, when the backend is built."""
        _parse_options(option_field_items(self))

    # ---- analyze / factor / solve hooks -------------------------------------
    def _analyze(
        self, topology: SparseTopology, options: tuple, layout: Layout, precision: str
    ) -> _SuperLUAnalysis:
        """Build the compressed-column structure and column ordering of ``topology``.

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
                the library cannot take, or a missing native extension.
        """
        del layout, precision  # analyze() already rejected multi-device layouts and fp32
        parsed = _parse_options(options)
        try:
            native = _native()
        except ImportError as exc:
            raise ConfigurationError(str(exc)) from exc
        n = int(topology.n_dofs)
        rows = np.ascontiguousarray(np.asarray(topology.rows), dtype=np.int32)
        cols = np.ascontiguousarray(np.asarray(topology.cols), dtype=np.int32)
        mask, diagonal = dirichlet_rows(topology)
        zeroed = np.zeros(rows.shape, dtype=np.bool_) if mask is None else np.asarray(mask)
        bc = topology.bc_indices
        if bc is not None and int(np.asarray(bc).size) != int(diagonal.size):
            missing = sorted(set(np.asarray(bc).ravel().tolist()) - set(diagonal.tolist()))
            raise ConfigurationError(
                f"SuperLU needs a diagonal entry in every Dirichlet row; rows {missing} have "
                "none in the pattern. Add the diagonal entries to the topology."
            )
        try:
            native_id = native.analysis_create(
                n,
                rows,
                cols,
                np.ascontiguousarray(zeroed, dtype=np.bool_),
                np.ascontiguousarray(diagonal, dtype=np.int32),
                ORDERINGS[parsed["ordering"]],
                parsed["panel_size"],
                parsed["relax"],
                parsed["pivot_threshold"],
            )
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        return _SuperLUAnalysis(native_id, n, int(rows.size), parsed["ordering"])

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
        keeps one slot per ``(context id, slot token)`` at that site,
        creating it on first use and refactoring it on later runs, so the
        lanes of a value batch (distinct slot tokens) get one slot each and
        every context gets its own. Both orientations factor ``A_bc``; a
        solve with orientation ``"AT"`` runs ``dgstrs`` transposed.

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

        Runs ``dgstrs`` with ``trans`` set when exactly one of
        ``orientation == "AT"`` and ``transpose`` holds. ``b`` of shape
        ``(nrhs, n)`` is solved as one block. The solution is ``float64``,
        the result dtype of the ``"fp64"`` rung, which ``b`` already has;
        it is ``nan`` unless the factorization succeeded. The call fails when
        the handle's context has been released or its slot was built for
        another analysis (the call carries the analysis's native id as
        ``analysis_id``). An empty handle (slot ``-1``) makes no call and
        returns ``nan`` with ``SolveStatus.EMPTY``.

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
        one of the site's slots is refactored in place (a loop carrying the
        factor keeps one slot, and the same pattern reuses its ordering,
        elimination tree and ``L``/``U`` storage); any other held slot gets
        the site's slot for it in the handle's context, so other holders of
        that factor keep it. A false predicate returns the held handle and
        that slot's status. The call fails when the held slot was built for
        another analysis. An empty handle (slot ``-1``) whose predicate
        holds is factored into the site's own slot, and one whose predicate
        is false stays empty with ``SolveStatus.EMPTY``. The kept values
        are ``values`` where the predicate holds and ``held_values``
        elsewhere. ``adopt`` is accepted without effect: this backend has
        one rule for a held slot of another site. A factor that names shared
        storage (``storage``) is refactored in place in that storage's slot,
        and with a false predicate only when the slot holds other values
        than the ones the factor keeps; an empty handle fills the storage's
        slot.

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
        """Bytes of the slots held for ``ctx``: values, permutations and the nonzeros of L and U."""
        return int(_native().resident_bytes(int(ctx.context_id)))

    # ---- diagnostics ----------------------------------------------------------
    @staticmethod
    def native_stats() -> dict:
        """Process-wide native counters.

        Returns:
            A dict with ``factorizations`` (fresh ``pdgstrf`` runs),
            ``refactorizations`` (``pdgstrf`` reusing a slot's structure),
            ``rebuilds`` (factorizations a solve started because its slot held
            other values), ``solves`` (``dgstrs`` calls), ``nonfinite``
            (factor requests with non-finite values),
            ``released_context_errors`` (calls refused because their
            context was released), ``live_slots``, ``live_sites`` and
            ``live_analyses``.
        """
        return dict(_native().stats())

    @staticmethod
    def reset_native_stats() -> None:
        """Zero the counters of [native_stats][diff_soup.linear.superlu_backend.SuperLU.native_stats]."""
        _native().reset_stats()

    @staticmethod
    def threads() -> int:
        """Threads each factorization uses: the CPU affinity count, capped by ``OMP_NUM_THREADS``."""
        return int(_native().nprocs())
