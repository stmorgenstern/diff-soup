"""The analysis record of a cuDSS factorization spread over a ``"model"`` mesh axis.

[analyze][diff_soup.linear.analyze] with ``backend=CuDSS()`` and a layout
whose ``"model"`` axis spreads one factorization over several devices builds
its record through
[analyze_hook][diff_soup.linear._cudss._model_axis._analysis.analyze_hook]:

- Within one process (cuDSS MG), a
  [ModelGroupAnalysis][diff_soup.linear._cudss._model_axis._analysis.ModelGroupAnalysis]:
  a native contract analysis, as on one device, whose factors each spread
  over the CUDA devices of one model group of the process. Its factors are
  held in a [SolveContext][diff_soup.linear.SolveContext], refactored in
  place and solved by the contract handlers of
  ``diff_soup.linear._cudss._contract``.
- Across processes (MGMN), a
  [ModelAxisAnalysis][diff_soup.linear._cudss._model_axis._analysis.ModelAxisAnalysis]:
  the CSR pattern of the operator, the cuDSS options of the native per-call
  solver and one native solver instance, which factors ``A_bc`` and solves
  with it or, in cuDSS's transpose solve mode, with its transpose; its
  pattern is registered with the extension under a content-stable id. That
  id is an attribute of every lowered solve, so a program compiled in one
  process has the same text, and the same persistent compilation-cache key,
  in every other process.

Importing this module loads the native cuDSS extension, so it fails on a host
without one.
"""

from __future__ import annotations

import dataclasses
import hashlib
import struct
import threading
import weakref
from typing import Any

import jax
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup._layout import Layout, linear_mode
from diff_soup.linear._api import SparseTopology, _freeze_options
from diff_soup.linear._cudss._contract import (
    CSRPattern,
    _CuDSSContractAnalysis,
    _digest,
    _is_int,
    _native,
    _NativeAnalysis,
    _parse_cudss_alg_id,
    _parse_options,
    _resolve_reordering_alg,
    csr_pattern,
)
from diff_soup.linear._cudss._ffi_loader import native_solver
from diff_soup.linear._cudss._model_axis._mgmn import prepare

__all__ = ["ModelAxisAnalysis", "ModelGroupAnalysis", "analyze_hook", "solver_instance_id"]

_STABLE_ID_PREFIX = 1 << 62
_STABLE_ID_MASK = (1 << 62) - 1


def _stable_key_bytes(value: Any) -> bytes:
    """Encode an immutable host value without process-local hashes.

    Native instance ids are integer attributes of the lowered program, so
    they must not depend on Python's salted ``hash`` or on ``repr`` of an
    arbitrary object: the persistent compilation-cache key would then change
    between processes. Only ``None``, bools, integers, floats, strings,
    bytes, tuples, lists and dataclass instances of those are encoded; any
    other value is refused.

    Raises:
        TypeError: For a value of any other type.
    """

    def framed(tag: bytes, payload: bytes) -> bytes:
        return tag + struct.pack(">Q", len(payload)) + payload

    if value is None:
        return b"N"
    if isinstance(value, (bool, np.bool_)):
        return b"B1" if bool(value) else b"B0"
    if isinstance(value, (int, np.integer)):
        return framed(b"I", str(int(value)).encode("ascii"))
    if isinstance(value, (float, np.floating)):
        return b"F" + struct.pack(">d", float(value))
    if isinstance(value, str):
        return framed(b"S", value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return framed(b"Y", bytes(value))
    if isinstance(value, tuple):
        return framed(b"T", b"".join(framed(b"E", _stable_key_bytes(item)) for item in value))
    if isinstance(value, list):
        return framed(b"L", b"".join(framed(b"E", _stable_key_bytes(item)) for item in value))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        type_name = f"{type(value).__module__}.{type(value).__qualname__}"
        fields = tuple(
            (field.name, getattr(value, field.name)) for field in dataclasses.fields(value)
        )
        return framed(b"D", _stable_key_bytes((type_name, fields)))
    raise TypeError(
        "cuDSS stable identities require immutable primitive/tuple/dataclass "
        f"keys; got {type(value).__module__}.{type(value).__qualname__}."
    )


_ID_LOCK = threading.Lock()
_DIGEST_BY_ID: dict[int, bytes] = {}


def _stable_id(namespace: str, key: Any) -> int:
    """A process-stable signed-int64 id for one logical resource.

    The namespace and the encoded key are digested with SHA-256; the id
    lies in the positive upper quarter of int64. A digest collision raises
    instead of being resolved by allocation order.
    """
    digest = hashlib.sha256(_stable_key_bytes((str(namespace), key))).digest()
    stable_id = _STABLE_ID_PREFIX | (int.from_bytes(digest[:8], "big") & _STABLE_ID_MASK)
    with _ID_LOCK:
        previous = _DIGEST_BY_ID.get(stable_id)
        if previous is not None and previous != digest:
            raise RuntimeError(
                "cuDSS deterministic identity collision; change the identity "
                "namespace/version rather than allocating by process order."
            )
        _DIGEST_BY_ID[stable_id] = digest
    return int(stable_id)


def solver_instance_id(instance_key: Any) -> int:
    """The content-stable native solver instance id of ``instance_key``.

    Args:
        instance_key: An immutable host value (see ``_stable_key_bytes``),
            here ``(content key, "primal")``.

    Returns:
        The id, the same in every process for an equal key.
    """
    return _stable_id("solver-v1", ("base-solver-v1", instance_key))


def _content_key(topology: SparseTopology, options: dict[str, object]) -> tuple:
    """The content identity of one pattern and set of solver options."""
    bc = topology.bc_indices
    return (
        "cudss-layer-v1",
        int(topology.n_dofs),
        np.asarray(topology.rows, dtype=np.int32).tobytes(),
        np.asarray(topology.cols, dtype=np.int32).tobytes(),
        None if bc is None else np.asarray(bc, dtype=np.int32).tobytes(),
        _freeze_options(options),
    )


def _optional_flag(value: object, *, default: int = -1) -> int:
    """A boolean-like cuDSS option as an int (``default`` when unset)."""
    if value is None:
        return int(default)
    return int(bool(value))


def _model_group_ordinals(mesh: Any) -> tuple[int, ...]:
    """The CUDA ordinals of every model group this process holds, group after group.

    Each group is in the mesh's model-axis order. cuDSS MG spreads a
    factorization over the group that holds the device a solve runs on, so
    under a ``shard_map`` over the data axis each shard factors on its own
    group. Each JAX device is mapped to its CUDA ordinal in this process
    (``Device.local_hardware_id``).

    Raises:
        ConfigurationError: If this process holds no model group of the mesh,
            or a device of its groups is not a CUDA device.
    """
    axis = mesh.axis_names.index("model")
    rows = np.moveaxis(mesh.devices, axis, -1).reshape((-1, mesh.shape["model"]))
    here = [row for row in rows if all(d.process_index == jax.process_index() for d in row)]
    if not here:
        raise ConfigurationError(
            "cuDSS MG factors over the devices of a model group of this process, and this "
            "process holds no model group of the layout's mesh whole."
        )
    devices = [device for row in here for device in row]
    if any(device.platform != "gpu" for device in devices):
        raise ConfigurationError(
            "cuDSS MG needs CUDA devices on its model axis; the model groups of this process "
            f"hold {[str(device) for device in devices]}."
        )
    return tuple(int(device.local_hardware_id) for device in devices)


#: The values of the ``mgmn_input`` option, by the integer the native solver takes.
_MGMN_INPUTS = ("central", "distributed")


def _mgmn_input(opts: dict[str, Any], mode: str) -> int:
    """The ``mgmn_input`` option as the native solver's integer (``0`` central, ``1`` distributed).

    Raises:
        ConfigurationError: For a value other than ``"central"`` or
            ``"distributed"``, or ``"distributed"`` on a model axis within one
            process (``"mg"``), which cuDSS cannot distribute.
    """
    value = opts.pop("mgmn_input", "central")
    if not isinstance(value, str) or value not in _MGMN_INPUTS:
        raise ConfigurationError(
            f"cuDSS option mgmn_input must be one of {list(_MGMN_INPUTS)}; got {value!r}."
        )
    if value == "distributed" and mode != "mgmn":
        raise ConfigurationError(
            "cuDSS option mgmn_input='distributed' needs a 'model' axis across processes "
            "(MGMN); this layout's model axis is within one process. Leave mgmn_input out "
            "or set it to 'central'."
        )
    return _MGMN_INPUTS.index(value)


def _solver_options(options: tuple, layout: Layout) -> dict[str, Any]:
    """The cuDSS options of an analysis across processes, with the layout's settings applied.

    A layout ``reorder`` other than the default ``"amd"`` is applied unless
    the options name one; ``layout.hybrid_memory`` sets ``hybrid_mode`` and
    the layout's device-memory limit. ``mgmn_group``, the id of the
    initialised multi-process communicator of the model group, is added, and
    ``mgmn_input="distributed"`` is kept as ``mgmn_input=1``; both are part
    of the content key, so different model groups and input modes get
    different native instances. ``"central"``, the default, is dropped.

    Raises:
        ConfigurationError: For deterministic mode or a custom device memory
            handler, or an invalid ``mgmn_input``.
    """
    opts = dict(options)
    if opts.get("ir_nsteps", 0) is None:
        del opts["ir_nsteps"]  # None is the option left out
    if _mgmn_input(opts, "mgmn"):
        opts["mgmn_input"] = 1
    if layout.reorder != "amd" and "reorder" not in opts:
        opts["reorder"] = layout.reorder
    if layout.hybrid_memory:
        opts["hybrid_mode"] = 1
        if layout.hybrid_device_memory_limit_mb is not None:
            opts["hybrid_device_memory_limit_mb"] = int(layout.hybrid_device_memory_limit_mb)
    if opts.get("deterministic") or opts.get("use_device_mem_handler"):
        raise ConfigurationError(
            "MGMN does not support deterministic mode or a custom memory handler."
        )
    opts["mgmn_group"] = prepare(layout.mesh)
    return opts


def _call_attributes(opts: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    """The integer FFI attributes of a solve, except the instance id, the refit mode and the device set.

    Without ``use_matching`` and ``matching_alg`` the solve runs cuDSS's
    matching with scaling (``use_matching=1``, which the extension sends as
    ``CUDSS_MATCHING_ALG_AUTO``), as a one-device factorization does; cuDSS
    turns it off for a symmetric positive definite ``mtype_id``.

    Without ``ir_nsteps`` (or with ``ir_nsteps=None``) the solve runs no
    iterative refinement (``ir_nsteps=0``), as a float64 factor kept in
    float64 does on one device: the per-call solver factors in float64,
    where cuDSS 1.0 wrote no stale output into warm solves without
    refinement. An explicit ``ir_nsteps`` is used as given.

    Raises:
        ConfigurationError: For deterministic mode with iterative
            refinement, an ``ir_nsteps`` that is not a non-negative integer,
            or an unknown ordering or algorithm name.
    """
    deterministic = _optional_flag(opts.get("deterministic"), default=0)
    ir_nsteps = opts.get("ir_nsteps")
    if ir_nsteps is None:
        ir_nsteps = 0
    elif not _is_int(ir_nsteps) or int(ir_nsteps) < 0:
        raise ConfigurationError(
            f"cuDSS option ir_nsteps must be None or a non-negative integer; got {ir_nsteps!r}."
        )
    ir_nsteps = int(ir_nsteps)
    if deterministic and ir_nsteps > 0:
        raise ConfigurationError(
            "cuDSS deterministic mode runs without iterative refinement, so it needs "
            f"ir_nsteps=0 or ir_nsteps left out; got ir_nsteps={ir_nsteps!r} with "
            "deterministic=True."
        )
    limit = opts.get("hybrid_device_memory_limit_mb")
    try:
        reordering_alg = _resolve_reordering_alg(opts)
        matching_alg = _parse_cudss_alg_id(opts.get("matching_alg"))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    attributes = {
        "device_id": int(opts.get("device_id", 0)),
        "mtype_id": int(opts.get("mtype_id", 0)),
        "mview_id": int(opts.get("mview_id", 0)),
        "ir_nsteps": ir_nsteps,
        "reordering_alg": int(reordering_alg),
        "use_matching": _optional_flag(
            opts.get("use_matching"),
            default=1 if opts.get("matching_alg") is None else -1,
        ),
        "matching_alg": int(matching_alg),
        "use_device_mem_handler": int(
            bool(_optional_flag(opts.get("use_device_mem_handler"), default=0))
        ),
        "use_superpanels": _optional_flag(opts.get("use_superpanels")),
        "hybrid_mode": int(bool(opts.get("hybrid_mode", 0))),
        "hybrid_device_memory_limit_mb": -1 if limit is None else int(limit),
        "deterministic": int(deterministic),
        "mg_group_size": int(opts.get("mg_group_size", 0)),
        "mgmn_group": int(opts.get("mgmn_group", 0)),
        "mgmn_input": int(opts.get("mgmn_input", 0)),
    }
    return tuple(sorted(attributes.items()))


class ModelAxisAnalysis:
    """Host record of a cuDSS analysis whose factorization spans a ``"model"`` axis across processes.

    Shared by every [Analysis][diff_soup.linear.Analysis] of one pattern and
    set of solver options (see
    [analyze_hook][diff_soup.linear._cudss._model_axis._analysis.analyze_hook]).
    Its factors run on the native per-call solver (MGMN).

    Attributes:
        mode: ``"mgmn"`` (one device in each of several processes).
        pattern: The [CSRPattern][diff_soup.linear._cudss._contract.CSRPattern]
            of the operator, with the map that assembles coordinate values
            into its CSR order.
        instance_id: The native solver instance that factors ``A_bc`` and
            solves with it or its transpose.
        attributes: The remaining integer FFI attributes of a solve, as
            sorted ``(name, value)`` pairs.
        devices: Empty: an MGMN solve names no device set of this process.
    """

    __slots__ = ("mode", "pattern", "instance_id", "attributes", "devices")

    def __init__(
        self,
        mode: str,
        pattern: CSRPattern,
        instance_id: int,
        attributes: tuple[tuple[str, int], ...],
        devices: tuple[int, ...] = (),
    ):
        """Record the pattern, the native instance, the solve attributes and the device set."""
        self.mode = mode
        self.pattern = pattern
        self.instance_id = int(instance_id)
        self.attributes = attributes
        self.devices = tuple(int(device) for device in devices)

    def __repr__(self) -> str:
        """Name the mode, the sizes and the instance id."""
        return (
            f"ModelAxisAnalysis(mode={self.mode!r}, n={self.pattern.n}, nnz={self.pattern.nnz}, "
            f"nse={self.pattern.nse}, instance_id={self.instance_id}, devices={self.devices})"
        )


_RECORDS: dict[tuple, ModelAxisAnalysis] = {}
_RECORDS_LOCK = threading.Lock()


class ModelGroupAnalysis(_CuDSSContractAnalysis):
    """Host record of a cuDSS analysis whose factors spread over a model group of one process.

    A native contract analysis, as on one device (see
    ``diff_soup.linear._cudss._contract``), whose every factor spreads over
    the CUDA devices of one model group of this process with cuDSS MG: the
    group that holds the device of the factor's first call, that device
    first. Its factors are held in a
    [SolveContext][diff_soup.linear.SolveContext], refactored in place and
    solved by the contract handlers, and counted by their
    ``contract_stats``. Shared by every [Analysis][diff_soup.linear.Analysis]
    of one pattern, set of options and device set; the native analysis is
    released when the last such ``Analysis`` is collected.

    Attributes:
        devices: The CUDA ordinals of the model groups this process holds,
            group after group, each in the mesh's order.
        group_size: The devices of one model group.
    """

    __slots__ = ("devices", "group_size")

    def __init__(
        self,
        native: _NativeAnalysis,
        pattern: CSRPattern,
        reorder: str,
        devices: tuple[int, ...],
        group_size: int,
    ):
        """Record the assembly maps of ``pattern`` over ``native`` and the model groups."""
        super().__init__(native, pattern, reorder)
        self.devices = tuple(int(device) for device in devices)
        self.group_size = int(group_size)

    def __repr__(self) -> str:
        """Name the native id, the sizes, the ordering and the model groups."""
        return (
            f"ModelGroupAnalysis(id={self.native_id}, n={self.n}, nnz={self.nnz}, "
            f"nse={self.nse}, reorder={self.reorder!r}, devices={self.devices}, "
            f"group_size={self.group_size})"
        )


#: The analyze options of a model axis within one process.
_MODEL_GROUP_OPTIONS = (
    "reorder",
    "hybrid_memory",
    "hybrid_device_memory_limit_mb",
    "ir_nsteps",
    "deterministic",
    "mgmn_input",
)


def _model_group_options(options: tuple, layout: Layout) -> dict[str, Any]:
    """The parsed contract options of an analysis over a model group of one process.

    Raises:
        ConfigurationError: For an option outside ``_MODEL_GROUP_OPTIONS``,
            deterministic mode, ``mgmn_input="distributed"``, or an invalid
            value.
    """
    opts = dict(options)
    _mgmn_input(opts, "mg")
    unknown = sorted(set(opts) - set(_MODEL_GROUP_OPTIONS))
    if unknown:
        raise ConfigurationError(
            f"cuDSS on a 'model' axis within one process (cuDSS MG) does not take the options "
            f"{unknown}; it accepts {list(_MODEL_GROUP_OPTIONS)}."
        )
    if opts.pop("deterministic", False):
        raise ConfigurationError(
            "options['deterministic'] (cuDSS DETERMINISTIC_MODE) is single-GPU only and cannot "
            "be combined with a multi-device model axis."
        )
    return _parse_options(tuple(sorted(opts.items())), layout)


#: Model-group records by pattern, options and device set. Weak values: an entry lives while
#: some [Analysis][diff_soup.linear.Analysis] holds its record.
_GROUP_RECORDS: "weakref.WeakValueDictionary[tuple, ModelGroupAnalysis]" = (
    weakref.WeakValueDictionary()
)


def _model_group_analysis(
    topology: SparseTopology, options: tuple, layout: Layout
) -> ModelGroupAnalysis:
    """The shared record of a cuDSS analysis over the model groups of this process.

    Raises:
        ConfigurationError: For an option a model group does not take, or a
            process that holds no model group of the layout's mesh, or one of
            other than CUDA devices.
    """
    parsed = _model_group_options(options, layout)
    devices = _model_group_ordinals(layout.mesh)
    group_size = int(layout.dof_size)
    pattern = csr_pattern(topology)
    key = (
        pattern.n,
        pattern.nse,
        pattern.nnz,
        _digest(
            pattern.offsets,
            pattern.columns,
            pattern.forward_index,
            pattern.zeroed,
            pattern.diagonal_forward,
        ),
        tuple(sorted(parsed.items())),
        devices,
        group_size,
    )
    with _RECORDS_LOCK:
        record = _GROUP_RECORDS.get(key)
        if record is not None:
            return record
        native = _native()
        try:
            native_id = native.contract_analysis_create(
                pattern.n,
                pattern.offsets,
                pattern.columns,
                pattern.nnz,
                pattern.gather_offsets,
                pattern.gather_entries,
                pattern.diagonal_add,
                parsed["reorder_alg"],
                parsed["hybrid_mode"],
                parsed["hybrid_device_memory_limit_mb"],
                parsed["deterministic"],
                parsed["ir_nsteps"],
                parsed["symmetric_factorization"],
                parsed["zero_fill"],
                parsed["downcast_factors"],
                parsed["emulation_mantissa_bits"],
            )
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        holder = _NativeAnalysis(native_id)
        native.contract_analysis_set_model_groups(
            holder.native_id, np.asarray(devices, dtype=np.int32), group_size
        )
        record = ModelGroupAnalysis(holder, pattern, parsed["reorder"], devices, group_size)
        _GROUP_RECORDS[key] = record
        return record


def _register_topology(instance_id: int, offsets: np.ndarray, columns: np.ndarray) -> None:
    """Hand a CSR pattern to the native extension under ``instance_id``.

    The extension keeps a host copy per instance id, so the solves of the
    instance carry empty pattern attributes and a compiled program's text
    depends on the pattern only through the id.
    """
    native_solver.register_topology(
        int(instance_id),
        np.ascontiguousarray(offsets, dtype=np.int32),
        np.ascontiguousarray(columns, dtype=np.int32),
    )


def analyze_hook(
    topology: SparseTopology, options: tuple, layout: Layout, precision: str
) -> ModelAxisAnalysis | ModelGroupAnalysis:
    """Build the record of a cuDSS analysis over the ``"model"`` axis of ``layout``.

    One record exists per pattern and set of solver options (and, within
    one process, per device set) in the process: later calls with equal
    content return it.

    Args:
        topology: The sparsity pattern.
        options: The frozen backend options. Within one process (cuDSS MG):
            ``reorder``, ``hybrid_memory``, ``hybrid_device_memory_limit_mb``
            and ``ir_nsteps``, as on one device (``deterministic`` and
            ``mgmn_input="distributed"`` are refused). Across processes
            (MGMN) every option of the native per-call solver: ``reorder``,
            ``reordering_alg``, ``hybrid_mode``,
            ``hybrid_device_memory_limit_mb``, ``deterministic``,
            ``ir_nsteps``, ``device_id``, ``mtype_id``, ``mview_id``,
            ``use_matching``, ``matching_alg``, ``use_device_mem_handler``
            and ``use_superpanels``, and ``mgmn_input`` (``"central"``, the
            default, or ``"distributed"``).
        layout: A layout whose ``"model"`` axis spreads one factorization.
        precision: The precision rung; only ``"fp64"`` is served.

    Returns:
        The shared record: a
        [ModelGroupAnalysis][diff_soup.linear._cudss._model_axis._analysis.ModelGroupAnalysis]
        within one process, a
        [ModelAxisAnalysis][diff_soup.linear._cudss._model_axis._analysis.ModelAxisAnalysis]
        across processes.

    Raises:
        ConfigurationError: For a rung other than ``"fp64"``, a layout that
            keeps the factorization on one device, an option the mode does
            not take, deterministic mode with a model axis or with iterative
            refinement, a custom memory handler across processes, an unknown
            ordering or algorithm name, or an invalid ``mgmn_input``.
    """
    mode = linear_mode(layout)
    if mode not in ("mg", "mgmn"):
        raise ConfigurationError(
            "The cuDSS model-axis analysis needs a layout whose 'model' axis spreads one "
            f"factorization over several devices; this layout factors on one device ({mode!r})."
        )
    if precision != "fp64":
        raise ConfigurationError(
            "cuDSS factors a model-axis layout in double precision only; got the precision "
            f"rung {precision!r}. Analyze at precision='fp64' (Precision.solve_mode='fp64')."
        )
    if mode == "mg":
        return _model_group_analysis(topology, options, layout)
    opts = _solver_options(options, layout)
    attributes = _call_attributes(opts)
    content_key = _content_key(topology, opts)
    with _RECORDS_LOCK:
        record = _RECORDS.get(content_key)
        if record is not None:
            return record
        pattern = csr_pattern(topology)
        instance_id = solver_instance_id((content_key, "primal"))
        _register_topology(instance_id, pattern.offsets, pattern.columns)
        record = ModelAxisAnalysis(mode, pattern, instance_id, attributes)
        _RECORDS[content_key] = record
        return record
