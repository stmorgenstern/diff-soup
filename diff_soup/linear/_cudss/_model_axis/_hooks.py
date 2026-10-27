"""The contract hooks of a cuDSS factorization spread over a ``"model"`` axis across processes.

A factor of a [ModelAxisAnalysis][diff_soup.linear._cudss._model_axis._analysis.ModelAxisAnalysis]
(MGMN: one device in each of several processes) holds no native storage of
its own: its handle is one ``int32`` marker (``-1`` for an empty factor,
``0`` for a filled one) and its values are the matrix. Every solve assembles
the factor's values into the CSR order of ``A_bc`` and makes one call of the
native per-call solver ``diffsoup_solve_single_f64`` on the analysis's one
native solver instance, which factors those values and solves
(``refit_mode=1``), with ``A_bc^T`` in cuDSS's transpose solve mode
(``solve_mode=1``) when the factor's orientation and the solve's
``transpose`` together apply the transpose. Every process of the model group
makes the call on its own device and cuDSS returns the whole solution on
every process; the call's status is ``SUCCESS``. In a program compiled over
several devices (arrays placed on the layout's mesh) the call runs under a
``jax.shard_map`` with every operand and result replicated, so each device
of the program makes it once with the whole operands and the compiled
program holds no reduction. Factorizations,
refactorizations and invalidations make no native call. A model axis within
one process (cuDSS MG) does not use these hooks: its factors are held by the
contract handlers (see
[ModelGroupAnalysis][diff_soup.linear._cudss._model_axis._analysis.ModelGroupAnalysis]).

The JAX-level functions here serve as ``factor_fun`` / ``apply_fun`` /
``refactor_fun`` of [CuDSS][diff_soup.linear._cudss._backend.CuDSS], and the
lowerings of its ``lower_*`` hooks lower them with ``mlir.lower_fun``.

Importing this module loads the native cuDSS extension and registers the
FFI target with JAX, so it fails on a host without the extension.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Callable, Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import ShapedArray
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from diff_soup._jax_compat import mlir
from diff_soup.linear._cudss._contract import assemble_csr
from diff_soup.linear._cudss._ffi_loader import native_solver, register_ffi_type_compat
from diff_soup.linear._cudss._model_axis._analysis import ModelAxisAnalysis
from diff_soup.linear._factor import (
    Analysis,
    FactorContractError,
    SolveStatus,
    effective_transpose,
)
from diff_soup.linear._primitive import _jax_level

__all__ = [
    "SOLVE_TARGET",
    "apply_fun",
    "empty_handle",
    "factor_avals",
    "factor_fun",
    "lowering",
    "model_axis_record",
    "refactor_fun",
    "solve_attributes",
]

#: The native per-call FFI target a model-axis solve binds.
SOLVE_TARGET = "diffsoup_solve_single_f64"

# The handler's state type is registered before the handler itself, as XLA
# requires of the FFI types a handler uses.
register_ffi_type_compat(SOLVE_TARGET, native_solver.type_id_f64(), native_solver.type_info_f64())
jax.ffi.register_ffi_target(SOLVE_TARGET, native_solver.handler_f64(), platform="CUDA")

#: The empty pattern attributes of a solve: the extension holds the pattern
#: per instance id.
_EMPTY_PATTERN = np.zeros((0,), dtype=np.int32)

_MARKER_DTYPE = np.dtype(np.int32)

#: The devices of the program a model-axis solve is being lowered into, while that program
#: spans more than one device and the solve is not already inside a ``shard_map``; else ``None``.
_PROGRAM_DEVICES: contextvars.ContextVar[Optional[tuple]] = contextvars.ContextVar(
    "diff_soup_cudss_model_axis_program_devices", default=None
)

#: The axis name of the mesh a solve's ``shard_map`` runs over.
_SOLVE_AXIS = "cudss_model_axis_solve"


def model_axis_record(analysis: Analysis) -> Optional[ModelAxisAnalysis]:
    """The record behind ``analysis`` of a model axis across processes, or ``None`` for any other."""
    record = analysis.backend_analysis
    return record if isinstance(record, ModelAxisAnalysis) else None


def _record(analysis: Analysis) -> ModelAxisAnalysis:
    """The model-axis record behind ``analysis``.

    Raises:
        FactorContractError: If the analysis keeps its factorization on one device.
    """
    record = model_axis_record(analysis)
    if record is None:
        raise FactorContractError(
            "The cuDSS model-axis hooks need an analysis of a layout whose 'model' axis "
            "spreads one factorization over the devices of several processes; got a record "
            f"{analysis.backend_analysis!r}."
        )
    return record


def _check_values_dtype(dtype: Any) -> None:
    """Refuse values that are not float32 or float64.

    Raises:
        FactorContractError: For any other dtype.
    """
    if np.dtype(dtype) not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise FactorContractError(
            "cuDSS on a model-axis layout takes float32 or float64 values (factored in "
            f"float64); got {np.dtype(dtype).name}. Cast the values before the call."
        )


def _refuse_values_batch(values: Any, caller: str) -> None:
    """Refuse values with a lane axis: one factorization spans the model group.

    Raises:
        FactorContractError: If ``values`` has more than one axis.
    """
    if values.ndim != 1:
        raise FactorContractError(
            f"cuDSS on a model-axis layout runs one factorization over the devices of the model "
            f"axis, so its {caller} takes one matrix, values of shape ({values.shape[-1]},); "
            f"got shape {list(values.shape)} from a jax.vmap over the operator values. Batch "
            "the right-hand sides only, or put independent matrices on the layout's data axis "
            "with one model group per process (a launch of one process per entry of the data "
            "axis)."
        )


def _values_status(values: Any) -> jax.Array:
    """``NONFINITE`` where some value is not finite, else ``SUCCESS`` (one per lane)."""
    finite = jnp.all(jnp.isfinite(values), axis=-1)
    return jnp.where(finite, int(SolveStatus.SUCCESS), int(SolveStatus.NONFINITE)).astype(jnp.int32)


def _status(values: Any, marker: Any, factor_status: Any = int(SolveStatus.SUCCESS)) -> jax.Array:
    """``EMPTY`` where the marker is negative, else the worse of the values' and the factor's status.

    ``factor_status`` is the status the native call returned; ``NONFINITE``
    values outrank it.
    """
    worst = jnp.maximum(_values_status(values), jnp.asarray(factor_status, jnp.int32))
    return jnp.where(marker < 0, int(SolveStatus.EMPTY), worst).astype(jnp.int32)


def factor_avals(
    analysis: Analysis, values_aval: Any, *, orientation: str, precision: str, tol: Any
) -> tuple:
    """The handle: one ``int32`` scalar marker.

    Raises:
        FactorContractError: If the values are not float32 or float64.
    """
    del analysis, orientation, precision, tol
    _check_values_dtype(values_aval.dtype)
    return (ShapedArray((), _MARKER_DTYPE),)


def empty_handle(
    analysis: Analysis,
    values_aval: Any,
    *,
    context_id: int,
    orientation: str,
    precision: str,
    tol: Any,
) -> tuple:
    """The handle of an empty factor: the marker ``-1``.

    Raises:
        FactorContractError: If the values are not float32 or float64.
    """
    del analysis, context_id, orientation, precision, tol
    _check_values_dtype(values_aval.dtype)
    return (jnp.full((), -1, _MARKER_DTYPE),)


def factor_fun(
    values: Any,
    token: Any,
    context_id: Any,
    *,
    order: Any,
    analysis: Analysis,
    orientation: str,
    precision: str,
    tol: Any,
) -> tuple:
    """A filled marker and the status of ``values``; the factorization runs in each solve.

    No native call is made here, so a singular factor is reported by the
    solve that factors it.

    Returns:
        ``(marker, status, order)``: the marker ``0``, ``NONFINITE`` where a
        value is not finite (else ``SUCCESS``), and the order word unchanged.

    Raises:
        FactorContractError: For values with a lane axis.
    """
    del token, context_id, orientation, precision, tol
    _record(analysis)
    _refuse_values_batch(values, "factorization")
    return jnp.zeros((), _MARKER_DTYPE), _values_status(values), order


def solve_attributes(record: ModelAxisAnalysis, *, transposed: bool) -> dict[str, Any]:
    """The FFI attributes of one solve with ``A_bc`` or its transpose.

    Args:
        record: The model-axis record.
        transposed: Solve with ``A_bc^T`` (cuDSS's transpose solve mode on
            the factorization of ``A_bc``) instead of ``A_bc``.

    Returns:
        The empty pattern attributes, the instance id, ``refit_mode=1``,
        ``solve_mode`` (``1`` for the transpose, else ``0``) and the
        record's options, each an integer, and ``mg_devices``, an empty
        ``int32`` array (the per-call solver's device set of cuDSS MG,
        unused across processes).
    """
    return {
        "csr_offsets": _EMPTY_PATTERN,
        "csr_columns": _EMPTY_PATTERN,
        "instance_id": int(record.instance_id),
        "refit_mode": 1,
        "solve_mode": int(bool(transposed)),
        "mg_devices": np.asarray(record.devices, dtype=np.int32),
        **{name: int(value) for name, value in record.attributes},
    }


def _native_solve(
    analysis: Analysis, csr: Any, b: Any, attributes: dict[str, Any]
) -> tuple[jax.Array, jax.Array]:
    """One call of the per-call solver on the device the solve runs on.

    Returns the solution and the factor's ``int32`` status (``SUCCESS``).
    Every process makes the call on its own device with the whole operands,
    and cuDSS writes the whole solution on every process of the model group,
    so the result needs no reduction and the arrays need no placement on the
    layout's mesh. Lowered into a program over several devices, the call
    runs under a ``jax.shard_map`` over those devices with every operand and
    result replicated (see ``_solve_mesh``).
    """
    call = jax.ffi.ffi_call(
        SOLVE_TARGET,
        (jax.ShapeDtypeStruct(b.shape, b.dtype), jax.ShapeDtypeStruct((), jnp.int32)),
        has_side_effect=True,
    )
    mesh = _solve_mesh(analysis)
    if mesh is None:
        return call(b, csr, **attributes)
    # XLA's SPMD partitioner needs a sharding on every side-effecting call of a program over
    # several devices. Under the shard_map the call is manual on every device and each device
    # takes the whole (replicated) operands; its results are the same on every device, so they
    # leave replicated, and with varying-axis checking on no reduction is inserted.
    return jax.shard_map(
        lambda b, csr: call(b, csr, **attributes),
        mesh=mesh,
        in_specs=(P(), P()),
        out_specs=(P(), P()),
    )(b, csr)


def _solve_mesh(analysis: Analysis) -> Optional[Mesh]:
    """The mesh of the ``shard_map`` a solve runs under, or ``None`` to make the call as it is.

    ``None`` unless the solve is lowered into a program over several
    devices; then the layout's mesh when it holds exactly the program's
    devices, else a one-axis mesh over the program's devices (or the
    layout's mesh when the program names none).
    """
    devices = _PROGRAM_DEVICES.get()
    if devices is None:
        return None
    layout_mesh = analysis.layout.mesh
    if not devices or set(layout_mesh.devices.flat) == set(devices):
        return layout_mesh
    return Mesh(np.asarray(devices, dtype=object), (_SOLVE_AXIS,))


def _spans_devices(ctx: Any) -> Optional[tuple]:
    """The program's devices when ``ctx`` lowers into a program over several devices.

    ``None`` for a program on one device and inside a ``shard_map`` (whose
    body is already manual); an empty tuple when the program spans several
    devices but names none.
    """
    axis_context = ctx.module_context.axis_context
    if getattr(axis_context, "manual_axes", None) is not None:
        return None
    if int(getattr(axis_context, "num_devices", 1)) <= 1:
        return None
    return tuple(getattr(axis_context, "device_assignment", None) or ())


@contextlib.contextmanager
def _lowering_into(ctx: Any) -> Iterator[None]:
    """Record the devices of the program ``ctx`` lowers into for the solves lowered meanwhile."""
    token = _PROGRAM_DEVICES.set(_spans_devices(ctx))
    try:
        yield
    finally:
        _PROGRAM_DEVICES.reset(token)


def apply_fun(
    values: Any,
    marker: Any,
    *,
    order: Any,
    b: Any,
    analysis: Analysis,
    orientation: str,
    transpose: bool,
    precision: str,
    tol: Any,
) -> tuple:
    """Factor the factor's values on the analysis's native instance and solve.

    The values are assembled into the CSR values of ``A_bc`` in float64, and
    one native call refactors and solves, with ``A_bc^T`` in cuDSS's
    transpose solve mode when the factor's orientation and ``transpose``
    together apply the transpose. ``b`` of shape ``(n,)`` or ``(nrhs, n)`` goes to the call as it
    is. An empty factor makes no native call. The status is the worse of the
    values' status (``NONFINITE`` for a value that is not finite) and the
    status the native call returns (``SUCCESS`` across processes). The
    solution is ``nan`` unless the status is ``SUCCESS``.

    Returns:
        ``(x, status, order)`` with the order word unchanged.

    Raises:
        FactorContractError: For values with a lane axis.
    """
    del precision, tol
    record = _record(analysis)
    _refuse_values_batch(values, "solve")
    transposed = effective_transpose(orientation, transpose)
    csr = assemble_csr(record.pattern, values).astype(jnp.float64)
    attributes = solve_attributes(record, transposed=transposed)
    rhs = jnp.asarray(b)
    x, factor_status = jax.lax.cond(
        marker < 0,
        lambda operands: (
            jnp.zeros_like(operands[1]),
            jnp.asarray(int(SolveStatus.SUCCESS), jnp.int32),
        ),
        lambda operands: _native_solve(analysis, operands[0], operands[1], attributes),
        (csr, rhs),
    )
    status = _status(values, marker, factor_status)
    x = jnp.where(status == int(SolveStatus.SUCCESS), x, jnp.nan).astype(rhs.dtype)
    return x, status, order


def refactor_fun(
    values: Any,
    held_values: Any,
    predicate: Any,
    marker: Any,
    *,
    order: Any,
    analysis: Analysis,
    orientation: str,
    precision: str,
    tol: Any,
    adopt: bool,
) -> tuple:
    """Keep ``values`` where ``predicate`` holds and the held values elsewhere.

    No native call: the next solve factors the kept values and reports a
    singular factor. A lane whose predicate holds is filled (marker ``0``);
    a lane whose predicate is false keeps its marker, so an empty factor
    stays empty. Accepts one leading lane axis; ``adopt`` has no storage to
    select.

    Returns:
        ``(kept_values, marker, status, order)`` with the order word unchanged.
    """
    del orientation, precision, tol, adopt
    _record(analysis)
    predicate = jnp.asarray(predicate, jnp.bool_)
    kept = jnp.where(predicate[..., None], values, held_values)
    marker = jnp.where(predicate, 0, marker).astype(_MARKER_DTYPE)
    return kept, marker, _status(kept, marker), order


def lowering(kind: str) -> Callable[..., list]:
    """The MLIR lowering of the contract primitive ``kind`` for a model-axis analysis.

    Lowers the backend's JAX-level function for ``kind`` (``"factor"``,
    ``"apply"`` or ``"refactor"``) with ``mlir.lower_fun``, called with the
    primitive's operands and parameters as the contract's own lowering
    passes them; an ``"invalidate"`` lowers to the identity on the order
    word. A solve lowered into a program over several devices runs its
    native call under a replicated ``jax.shard_map`` (see ``_native_solve``).
    """

    def lower(ctx: Any, *operands: Any, **params: Any) -> list:
        fun = _jax_level(kind, ctx.avals_out)
        with _lowering_into(ctx):
            # mlir.lower_fun returns a tuple of MLIR values; the hook returns them as a list.
            return list(mlir.lower_fun(fun, multiple_results=True)(ctx, *operands, **params))

    return lower
