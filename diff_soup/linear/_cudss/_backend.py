"""cuDSS-backed [LinearSolver][diff_soup.linear.LinearSolver] implementation.

[CuDSS][diff_soup.linear._cudss._backend.CuDSS] serves the
[analyze][diff_soup.linear.analyze] / [factor][diff_soup.linear.factor] /
[solve][diff_soup.linear.solve] contract. On one CUDA device, and on a
``"model"`` axis within one process (cuDSS MG, each factor spread over the
devices of one model group), it runs through the contract handlers of the
native extension (``_contract.py``). A ``"model"`` axis across processes
(MGMN) runs through the hooks of the ``_model_axis`` package, which call the
native per-call solver once per solve, in double precision.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import equinox as eqx
from jaxtyping import Array

from diff_soup._errors import ConfigurationError
from diff_soup._layout import Layout, linear_mode
from diff_soup.linear._api import LinearSolver, SparseTopology, option_field_items
from diff_soup.linear._context import SolveContext
from diff_soup.linear._cudss import _contract
from diff_soup.linear._cudss._model_axis import _analysis as _model_axis_analysis
from diff_soup.linear._cudss._model_axis import _hooks as _model_axis
from diff_soup.linear._factor import Analysis, FactorContractError

#: Analyze options of the one-device route that a model axis does not take.
_ONE_DEVICE_OPTIONS = ("zero_fill", "downcast_factors", "emulation_mantissa_bits", "batch_layout")

_LOWER_FACTOR = _model_axis.lowering("factor")
_LOWER_APPLY = _model_axis.lowering("apply")
_LOWER_REFACTOR = _model_axis.lowering("refactor")
_LOWER_INVALIDATE = _model_axis.lowering("invalidate")


class CuDSS(LinearSolver):
    """NVIDIA cuDSS, a sparse direct solver on CUDA GPUs.

    It factors float64 and float32 matrices. `linear="cudss"` selects it with default settings,
    and `"auto"` picks it on a GPU. `ds.linear.CuDSS` is `None` when the cuDSS extension is
    not installed.

    Args:
        reorder: The fill-reducing ordering, `"amd"` or `"metis"`, where `None` takes the
            layout's `reorder`.
        hybrid_memory: `True` keeps part of the factor in host memory when it outgrows the GPU,
            and `None` takes the layout's.
        hybrid_device_memory_limit_mb: The GPU memory a hybrid factor may use, in MB, which
            needs `hybrid_memory=True`.
        deterministic: `True` runs cuDSS's deterministic mode, which allows no refinement,
            downcast or symmetric factorization.
        ir_nsteps: The iterative refinement steps per solve, where `None` takes 0 for a float64
            factor and 1 otherwise.
        symmetric_factorization: `True` factors an analysis declared `symmetric` with cuDSS's
            symmetric factorization instead of a general LU.
        zero_fill: `True` zeros the solution buffer before every solve, and `None` does so for
            float32 factors only.
        downcast_factors: `True` or `"block_diagonal"` stores the factor in float32 except its
            diagonal blocks, and `"strict_diagonal"` except its diagonal.
        emulation_mantissa_bits: From 1 to 64, the mantissa bits cuDSS uses to emulate a float64
            factorization on GPUs that support it.
        batch_layout: `"stacked"` (default) factors a `jax.vmap` batch as one block-diagonal
            matrix, and `"uniform"` refactors only the matrices that ask.

    Raises:
        ConfigurationError: For a setting outside these values, or a combination cuDSS refuses.
    """

    reorder: Optional[str] = eqx.field(static=True, default=None)
    hybrid_memory: Optional[bool] = eqx.field(static=True, default=None)
    hybrid_device_memory_limit_mb: Optional[int] = eqx.field(static=True, default=None)
    deterministic: bool = eqx.field(static=True, default=False)
    ir_nsteps: Optional[int] = eqx.field(static=True, default=None)
    symmetric_factorization: bool = eqx.field(static=True, default=False)
    zero_fill: Optional[bool] = eqx.field(static=True, default=None)
    downcast_factors: Union[None, bool, str] = eqx.field(static=True, default=None)
    emulation_mantissa_bits: Optional[int] = eqx.field(static=True, default=None)
    batch_layout: str = eqx.field(static=True, default="stacked")

    backend_name = "cudss"
    # The settings: every analyze option of the one-device route. The
    # options of the per-call solver of a model axis across processes stay
    # analyze options only.
    option_fields = _contract._OPTION_NAMES
    # Layout capabilities. mg_direct: cuDSS MG mode, a single-process
    # multi-GPU factorization (cudssCreateMg) selected by
    # ``Layout(mesh=<mesh with a local model group>)``, held by the contract
    # handlers; mgmn: one factorization over the devices of several processes,
    # on the model-axis hooks.
    supports_hybrid_memory = True
    supports_reordering = True
    supports_mg_direct = True
    supports_mgmn = True

    # analyze / factor / solve contract (see the class docstring).
    factor_residency = "device"
    transpose_is_free = True
    vmap_factors = "native"
    supported_precisions = ("fp64", "fp32")
    is_exact = True
    accepts_operator = False

    def __check_init__(self) -> None:
        """Reject a setting ``analyze`` would refuse, when the backend is built.

        A device-memory limit with ``hybrid_memory`` left out is checked
        against a layout with hybrid memory on, since the layout of a later
        solve may turn it on.
        """
        layout = Layout(hybrid_memory=self.hybrid_memory is None)
        _contract._parse_options(option_field_items(self), layout)

    # ---- analyze / factor / solve contract hooks -----------------------------
    # Each hook serves the contract route (``_contract``: one device, and a
    # "model" axis within one process) or, for an analysis of a "model" axis
    # across processes, the model-axis route (``_model_axis``), by the kind of
    # the analysis record.
    def _analyze(
        self, topology: SparseTopology, options: tuple, layout: Layout, precision: str
    ) -> Any:
        """Build the CSR patterns of ``topology`` and its cuDSS analysis record.

        A layout that keeps the factorization on one device gets the record
        of a native contract analysis; a layout whose ``"model"`` axis
        spreads it over the devices of one process gets a native contract
        analysis whose factors spread over a model group, and one whose
        ``"model"`` axis spans processes the record of a native per-call
        solver instance (see the class docstring).

        Args:
            topology: The sparsity pattern; its Dirichlet rows have their
                diagonal entry in the pattern.
            options: Frozen backend options (see the class docstring).
            layout: The validated layout.
            precision: The validated precision rung, ``"fp64"`` or ``"fp32"``
                (``"fp64"`` only on a model axis).

        Returns:
            The host record of the analysis.

        Raises:
            ConfigurationError: For an unknown or invalid option, a rung
                other than ``"fp64"`` on a model axis, or a native extension
                that is missing or built without the contract handlers.
        """
        if linear_mode(layout) != "local":
            # A model axis keeps its factors in float64, so downcast_factors at None or False
            # asks for what it does anyway.
            options = tuple(
                (k, v)
                for k, v in options
                if not (k == "downcast_factors" and (v is None or v is False))
            )
            local_only = sorted(set(dict(options)) & set(_ONE_DEVICE_OPTIONS))
            if local_only:
                raise ConfigurationError(
                    f"cuDSS options {local_only} apply to factorizations on one device; the "
                    "layout spreads the factorization over a 'model' mesh axis. Drop them or "
                    "pass a layout without a 'model' axis."
                )
            # An ir_nsteps of None (CuDSSOptions' default) is an option left out.
            options = tuple((k, v) for k, v in options if not (k == "ir_nsteps" and v is None))
            return _model_axis_analysis.analyze_hook(topology, options, layout, precision)
        return _contract.analyze_hook(topology, options, layout, precision)

    def factor_avals(
        self, analysis: Analysis, values_aval: Any, *, orientation: str, precision: str, tol: Any
    ) -> tuple:
        """The handle: the context id, the slot and its generation (``int32`` scalars).

        On a model axis across processes the handle is one ``int32`` scalar marker.

        Raises:
            FactorContractError: If the values are not real.
        """
        route = _model_axis if _model_axis.model_axis_record(analysis) is not None else _contract
        return route.factor_avals(
            analysis, values_aval, orientation=orientation, precision=precision, tol=tol
        )

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
        """The handle of an empty factor: the context id, slot ``-1`` and generation ``-1``.

        On a model axis across processes it is the marker ``-1``.

        Raises:
            FactorContractError: If the values are not real.
        """
        route = _model_axis if _model_axis.model_axis_record(analysis) is not None else _contract
        return route.empty_handle(
            analysis,
            values_aval,
            context_id=context_id,
            orientation=orientation,
            precision=precision,
            tol=tol,
        )

    def factor_fun(self, values: Array, token: Array, context_id: Array, **params: Any) -> tuple:
        """Record a model-axis factor's values and their status; no native call.

        Lowered by [lower_factor][diff_soup.linear._cudss._backend.CuDSS.lower_factor]
        for an analysis on a model axis across processes (see
        [factor_fun][diff_soup.linear._cudss._model_axis._hooks.factor_fun]).

        Raises:
            FactorContractError: For values with a lane axis, or an analysis
                whose factors the contract handlers hold (on one device or on
                a model axis within one process), which lowers to a native
                contract call.
        """
        return _model_axis.factor_fun(values, token, context_id, **params)

    def apply_fun(self, values: Array, *handle: Array, **params: Any) -> tuple:
        """Factor a model-axis factor's values and solve, in one native call.

        Lowered by [lower_apply][diff_soup.linear._cudss._backend.CuDSS.lower_apply]
        for an analysis on a model axis across processes (see
        [apply_fun][diff_soup.linear._cudss._model_axis._hooks.apply_fun]).

        Raises:
            FactorContractError: For values with a lane axis, or an analysis
                whose factors the contract handlers hold.
        """
        return _model_axis.apply_fun(values, *handle, **params)

    def refactor_fun(
        self, values: Array, held_values: Array, predicate: Array, *handle: Array, **params: Any
    ) -> tuple:
        """Keep a model-axis factor's new values where ``predicate`` holds; no native call.

        Lowered by [lower_refactor][diff_soup.linear._cudss._backend.CuDSS.lower_refactor]
        for an analysis on a model axis across processes (see
        [refactor_fun][diff_soup.linear._cudss._model_axis._hooks.refactor_fun]).

        Raises:
            FactorContractError: For an analysis whose factors the contract
                handlers hold.
        """
        return _model_axis.refactor_fun(values, held_values, predicate, *handle, **params)

    def lower_factor(self, ctx: Any, *operands: Any, **params: Any) -> list:
        """Lower a factorization to the CSR assembly and one FFI call ordered by the order word.

        The operands are ``(values, token, context_id, order)``. Each
        lowered call names a native site by a key computed from the lowered
        call, created on its first run. The handler keeps one slot per
        ``(context id, slot token)`` at that site (a batch slot for values
        with a lane axis) and factors ``A_bc`` (orientation ``"A"``) or
        ``A_bc^T`` (``"AT"``) at the rung's precision. A context id and slot
        token that are program constants travel as call attributes. On a
        model axis across processes it lowers ``factor_fun`` and emits no
        native call.

        Returns:
            The MLIR values of ``(context id, slot, generation, status, order)``
            (``(marker, status, order)`` on a model axis across processes).
        """
        if _model_axis.model_axis_record(params["analysis"]) is not None:
            return _LOWER_FACTOR(ctx, *operands, **params)
        return _contract.lower_factor(ctx, *operands, **params)

    def lower_apply(self, ctx: Any, *operands: Any, **params: Any) -> list:
        """Lower a solve to the CSR assembly and one FFI call ordered by the order word.

        The operands are ``(values, *handle, order, b)``. ``transpose=True``
        solves with the transpose of the slot's factorization in cuDSS's
        transpose solve mode, a native batch for every lane at once (an
        analysis declared ``symmetric`` or ``spd`` solves in normal mode).
        ``b`` of shape ``(nrhs, n)`` is solved as
        one block; the solution is ``nan`` unless the factorization
        succeeded. A batched solve takes ``b`` of shape ``(lanes, n)`` or
        ``(lanes, nrhs, n)``. A handle that comes straight from a factor or
        refactor call with a static identity travels as call attributes, so
        the handler need not read it back. A solve with ``transpose=False``
        of a factor that names shared storage (``storage``) uses that
        storage as it stands when a refactor of it ran earlier in the same
        execution. On a model axis across processes it lowers ``apply_fun``: one call of the
        native per-call solver that factors the factor's values and solves.

        Returns:
            The MLIR values of ``(x, status, order)``.

        Raises:
            FactorContractError: If ``b`` is complex.
        """
        if _model_axis.model_axis_record(params["analysis"]) is not None:
            return _LOWER_APPLY(ctx, *operands, **params)
        return _contract.lower_apply(ctx, *operands, **params)

    def lower_refactor(self, ctx: Any, *operands: Any, **params: Any) -> list:
        """Lower a refactorization where ``predicate`` holds to one ordered FFI call.

        The operands are ``(values, held_values, predicate, *handle, order)``.
        A held slot that is the call site's own is refactored in place, and
        so is any held slot when ``adopt`` is set; otherwise another site's
        held slot, and an empty handle, gets the site's slot for it in the
        handle's context. A false predicate returns the held handle with
        that slot's status. The call returns the kept values (``values``
        where the predicate holds, ``held_values`` elsewhere) first. A held
        handle that comes straight from a factor or refactor call with a
        static identity travels as call attributes, so the handler reads
        only the predicate back. A factor that names shared storage
        (``storage``) is refactored in that storage, which is made to hold
        the values the result keeps. On a model axis across processes it lowers
        ``refactor_fun`` and emits no native call.

        Returns:
            The MLIR values of ``(kept values, context id, slot, generation, status, order)``
            (``(kept values, marker, status, order)`` on a model axis across processes).
        """
        if _model_axis.model_axis_record(params["analysis"]) is not None:
            return _LOWER_REFACTOR(ctx, *operands, **params)
        return _contract.lower_refactor(ctx, *operands, **params)

    def lower_invalidate(self, ctx: Any, *operands: Any, **params: Any) -> list:
        """Lower an invalidation to one FFI call ordered by the order word.

        The operands are ``(*handle, context_id, order)``. The slot the
        handle names (every lane of a batch slot), or the shared storage a
        factor with ``storage`` names, no longer stands for any factor: the
        next solve on it refactors it from the values that solve carries. On
        a model axis across processes, whose every solve factors its own values, it is the
        identity on the order word.

        Returns:
            The MLIR value of the order word.
        """
        if _model_axis.model_axis_record(params["analysis"]) is not None:
            return _LOWER_INVALIDATE(ctx, *operands, **params)
        return _contract.lower_invalidate(ctx, *operands, **params)

    def declare_storage(self, analysis: Analysis, context_id: int, storage: str) -> None:
        """Record that a factor of the context names the shared storage ``storage``.

        Called by [empty_factor][diff_soup.linear.empty_factor] for
        ``storage=...``: a solve, refactor or invalidation of such a factor
        whose context id is not a program constant (a loop carry) then finds
        its storage without reading the handle while only one open context
        has declared that name on the analysis. A factor on a model axis across processes has
        no native storage, so nothing is recorded for it.
        """
        if _model_axis.model_axis_record(analysis) is not None:
            return
        _contract.declare_storage(analysis, context_id, storage)

    def _release(self, ctx: SolveContext) -> None:
        """Free every slot held for ``ctx`` (batch slots included) and record its id as released."""
        _contract.release(ctx)

    def _resident_bytes(self, ctx: SolveContext) -> int:
        """Device bytes of the slots held for ``ctx``: CSR value copies and cuDSS factors."""
        return _contract.resident_bytes(ctx)

    @staticmethod
    def contract_stats() -> dict:
        """Process-wide counters of the cuDSS contract handlers.

        Returns:
            The counters and ``live_*`` gauges described in
            [contract_stats][diff_soup.linear._cudss._contract.contract_stats].

        Raises:
            ConfigurationError: If the extension or its contract handlers are missing.
        """
        return _contract.contract_stats()

    @staticmethod
    def reset_contract_stats() -> None:
        """Zero the counters of [contract_stats][diff_soup.linear._cudss._backend.CuDSS.contract_stats].

        Raises:
            ConfigurationError: If the extension or its contract handlers are missing.
        """
        _contract.reset_contract_stats()

    @staticmethod
    def contract_analysis_info(analysis: Analysis) -> dict:
        """The native state of one cuDSS analysis (residency, reorderings, fill, timings).

        Args:
            analysis: A cuDSS [Analysis][diff_soup.linear.Analysis].

        Returns:
            The dict described in
            [contract_analysis_info][diff_soup.linear._cudss._contract.contract_analysis_info].

        Raises:
            ConfigurationError: If the extension or its contract handlers are missing.
            FactorContractError: If ``analysis`` is not a cuDSS analysis, or is one
                of a layout whose ``"model"`` axis spreads the factorization
                across processes, which has no native contract analysis.
        """
        if isinstance(analysis, Analysis) and _model_axis.model_axis_record(analysis) is not None:
            raise FactorContractError(
                "contract_analysis_info reports the native contract analysis of a cuDSS analysis "
                "on one device or on a 'model' axis within one process; this analysis spreads "
                "its factorization over a 'model' axis across processes and has none. Pass an "
                "analysis of a layout whose 'model' axis stays within one process."
            )
        return _contract.contract_analysis_info(analysis)

    @staticmethod
    def set_analysis_seeding(enabled: bool) -> bool:
        """Choose whether later analyses without matching reuse their pattern's first ordering.

        Symmetric positive definite slots and the lanes of native batches run
        without matching; general (and symmetric indefinite) slots always run
        their own analysis (their ordering depends on the values through
        cuDSS's matching); see
        [set_analysis_seeding][diff_soup.linear._cudss._contract.set_analysis_seeding].

        Args:
            enabled: Seed analyses without matching from the pattern's first ordering
                (the default).

        Returns:
            The previous setting.

        Raises:
            ConfigurationError: If the extension or its contract handlers are missing.
        """
        return _contract.set_analysis_seeding(enabled)


# Public alias used by downstream consumers and the cuDSS reference cases.
CuDSSBackend = CuDSS

__all__ = ["CuDSS", "CuDSSBackend"]
