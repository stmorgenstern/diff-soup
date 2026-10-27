"""Deterministic content fingerprints for inference declarations.

UQ result provenance must distinguish experiments by executable content, not
only by user-facing labels.  This module therefore encodes concrete arrays by
shape, dtype, and bytes and encodes static Python structure without falling
back to ``repr`` (whose default form commonly embeds a process-local address).

Unsupported or cyclic objects are rejected with their declaration path.  A
provenance hash that silently omits an opaque object is worse than no hash: it
can make a fit from a different statistical problem appear compatible.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import json
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AbstractMesh, Mesh, PartitionSpec

from diff_soup.problem.implicit import get_or_build_solver_view


class UnstableFingerprintError(TypeError):
    """A declaration contains an object with no stable encoding contract."""


def _qualified_type(value: Any) -> str:
    """Fully qualified ``module.QualName`` of ``value`` (or of its type)."""
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _canonical_json(value: Any) -> str:
    """Serialize ``value`` as compact, key-sorted JSON, rejecting NaN."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class _Recorder:
    """Convert supported objects to deterministic JSON-compatible records."""

    def __init__(self) -> None:
        self._memo: dict[int, Any] = {}
        self._active: dict[int, str] = {}
        # The memo is keyed by object identity, so every memoized value stays
        # referenced for the recorder's lifetime: a freed object's address can be
        # handed to a different object, whose record would otherwise be the
        # stale one stored under that address.
        self._alive: list[Any] = []

    def record(self, value: Any, *, path: str) -> Any:  # noqa: C901, PLR0911
        """Encode one value or raise with the exact unsupported-object path."""
        if value is None:
            return {"kind": "none"}
        if value is Ellipsis:
            return {"kind": "ellipsis"}
        if isinstance(value, bool):
            return {"kind": "bool", "value": value}
        if isinstance(value, int) and not isinstance(value, enum.Enum):
            return {"kind": "int", "value": str(value)}
        if isinstance(value, float):
            return {"kind": "float", "value": value.hex()}
        if isinstance(value, complex):
            return {
                "kind": "complex",
                "real": float(value.real).hex(),
                "imag": float(value.imag).hex(),
            }
        if isinstance(value, str):
            return {"kind": "str", "value": value}
        if isinstance(value, bytes):
            return {
                "kind": "bytes",
                "length": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
        if isinstance(value, enum.Enum):
            return {
                "kind": "enum",
                "type": _qualified_type(value),
                "name": value.name,
            }
        if isinstance(value, np.dtype):
            return {"kind": "dtype", "value": value.str}
        if isinstance(value, np.generic):
            return self._array_record(np.asarray(value), path=path)
        if isinstance(value, (np.ndarray, jax.Array)):
            return self._array_record(value, path=path)
        if isinstance(value, jax.ShapeDtypeStruct):
            return {
                "kind": "shape_dtype_struct",
                "shape": tuple(int(size) for size in value.shape),
                "dtype": np.dtype(value.dtype).str,
                "weak_type": bool(value.weak_type),
            }
        if isinstance(value, Path):
            return {"kind": "path", "value": value.as_posix()}
        if isinstance(value, type):
            return {"kind": "type", "value": _qualified_type(value)}
        if isinstance(value, types.ModuleType):
            return {"kind": "module", "value": value.__name__}
        if isinstance(value, range):
            return {
                "kind": "range",
                "start": value.start,
                "stop": value.stop,
                "step": value.step,
            }
        if isinstance(value, slice):
            return {
                "kind": "slice",
                "start": self.record(value.start, path=f"{path}.start"),
                "stop": self.record(value.stop, path=f"{path}.stop"),
                "step": self.record(value.step, path=f"{path}.step"),
            }
        if isinstance(value, types.CodeType):
            return self._code_record(value, path=path)
        if isinstance(value, functools.partial):
            return self._guarded(
                value,
                path=path,
                build=lambda: {
                    "kind": "partial",
                    "function": self.record(value.func, path=f"{path}.func"),
                    "args": self.record(value.args, path=f"{path}.args"),
                    "keywords": self.record(value.keywords or {}, path=f"{path}.keywords"),
                },
            )
        if isinstance(value, types.MethodType):
            return self._guarded(
                value,
                path=path,
                build=lambda: {
                    "kind": "bound_method",
                    "function": self.record(value.__func__, path=f"{path}.__func__"),
                    "self": self.record(value.__self__, path=f"{path}.__self__"),
                },
            )
        if isinstance(
            value,
            (types.FunctionType, types.BuiltinFunctionType, types.BuiltinMethodType),
        ):
            return self._function_record(value, path=path)
        if callable(value):
            # JAX exports activations as compiled or custom-derivative wrappers,
            # rather than Python functions. Their public identity and pinned
            # package version identify their implementation across processes.
            for namespace, module in (("jax.nn", jax.nn), ("jax.numpy", jnp)):
                names = sorted(
                    name
                    for name, member in vars(module).items()
                    if not name.startswith("_") and member is value
                )
                if names:
                    return {
                        "kind": "jax_function",
                        "namespace": namespace,
                        "name": names[0],
                        "version": jax.__version__,
                    }
        if isinstance(value, np.ufunc):
            return {
                "kind": "ufunc",
                "module": getattr(value, "__module__", "numpy"),
                "name": value.__name__,
                "nin": int(value.nin),
                "nout": int(value.nout),
            }
        if isinstance(value, (Mesh, AbstractMesh)):
            # A mesh is placement structure, not a value: the same declaration
            # fitted over a mesh of the same axis names, sizes, and kinds is the
            # same study whichever physical devices back it, and a fit that was
            # checkpointed on one node must resume on another. The device array
            # therefore stays out of the record.
            return {
                "kind": "mesh",
                "axis_names": [str(name) for name in value.axis_names],
                "axis_sizes": [int(value.shape[name]) for name in value.axis_names],
                "axis_types": [
                    getattr(kind, "name", str(kind)) for kind in getattr(value, "axis_types", ())
                ],
            }
        if isinstance(value, PartitionSpec):
            # PartitionSpec is not a tuple subclass any more; record its axis
            # assignments the way a tuple of axis names would be recorded.
            return {
                "kind": "partition_spec",
                "partitions": [
                    self.record(entry, path=f"{path}[{index}]") for index, entry in enumerate(value)
                ],
            }
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return self._guarded(
                value,
                path=path,
                build=lambda: {
                    "kind": "dataclass",
                    "type": _qualified_type(value),
                    "fields": [
                        {
                            "name": field.name,
                            "value": self.record(
                                getattr(value, field.name), path=f"{path}.{field.name}"
                            ),
                        }
                        for field in dataclasses.fields(value)
                        if field.metadata.get("fingerprint", True)
                    ],
                },
            )
        if isinstance(value, Mapping):
            return self._guarded(value, path=path, build=lambda: self._mapping(value, path))
        if isinstance(value, tuple):
            return self._guarded(
                value,
                path=path,
                build=lambda: {
                    "kind": "tuple",
                    "type": _qualified_type(value),
                    "items": [
                        self.record(item, path=f"{path}[{index}]")
                        for index, item in enumerate(value)
                    ],
                },
            )
        if isinstance(value, list):
            return self._guarded(
                value,
                path=path,
                build=lambda: {
                    "kind": "list",
                    "type": _qualified_type(value),
                    "items": [
                        self.record(item, path=f"{path}[{index}]")
                        for index, item in enumerate(value)
                    ],
                },
            )
        if isinstance(value, (set, frozenset)):
            return self._guarded(value, path=path, build=lambda: self._set(value, path))

        raise UnstableFingerprintError(
            f"Study fingerprint cannot stably encode {path}: unsupported "
            f"object type {_qualified_type(value)!r}. Use arrays, immutable "
            "dataclasses, plain containers, or a top-level/function closure "
            "composed only from those values."
        )

    def _guarded(self, value: Any, *, path: str, build: Any) -> Any:
        identity = id(value)
        if identity in self._memo:
            return self._memo[identity]
        if identity in self._active:
            raise UnstableFingerprintError(
                f"Study fingerprint cannot stably encode cyclic object at {path}; "
                f"it first appeared at {self._active[identity]}."
            )
        self._active[identity] = path
        try:
            record = build()
        finally:
            self._active.pop(identity, None)
        self._memo[identity] = record
        self._alive.append(value)
        return record

    def _array_record(self, value: Any, *, path: str) -> dict[str, Any]:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as exc:
            raise UnstableFingerprintError(
                f"Study fingerprint needs concrete host array content at {path}; "
                "traced or device-inaccessible values cannot define provenance."
            ) from exc
        if array.dtype.hasobject:
            raise UnstableFingerprintError(
                f"Study fingerprint cannot stably encode object-dtype array at {path}; "
                "its bytes contain process-local object pointers."
            )
        contiguous = np.ascontiguousarray(array)
        return {
            "kind": "array",
            "shape": tuple(int(size) for size in contiguous.shape),
            "dtype": contiguous.dtype.str,
            "weak_type": bool(getattr(value, "weak_type", False)),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }

    def _mapping(self, value: Mapping, path: str) -> dict[str, Any]:
        items = []
        for index, (key, item) in enumerate(value.items()):
            key_record = self.record(key, path=f"{path}.key[{index}]")
            item_record = self.record(item, path=f"{path}[key {index}]")
            items.append((_canonical_json(key_record), key_record, item_record))
        items.sort(key=lambda triple: triple[0])
        return {
            "kind": "mapping",
            "type": _qualified_type(value),
            "items": [
                {"key": key_record, "value": item_record} for _, key_record, item_record in items
            ],
        }

    def _set(self, value: set | frozenset, path: str) -> dict[str, Any]:
        records = [
            self.record(item, path=f"{path}.item[{index}]") for index, item in enumerate(value)
        ]
        records.sort(key=_canonical_json)
        return {
            "kind": "frozenset" if isinstance(value, frozenset) else "set",
            "type": _qualified_type(value),
            "items": records,
        }

    def _code_record(self, code: types.CodeType, *, path: str) -> dict[str, Any]:
        return {
            "kind": "code",
            "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
            "argcount": int(code.co_argcount),
            "posonlyargcount": int(code.co_posonlyargcount),
            "kwonlyargcount": int(code.co_kwonlyargcount),
            "flags": int(code.co_flags),
            "names": tuple(code.co_names),
            "varnames": tuple(code.co_varnames),
            "freevars": tuple(code.co_freevars),
            "cellvars": tuple(code.co_cellvars),
            "constants": self.record(code.co_consts, path=f"{path}.constants"),
        }

    def _function_record(self, value: Any, *, path: str) -> dict[str, Any]:
        seen: set[int] = set()
        while hasattr(value, "__wrapped__") and id(value) not in seen:
            seen.add(id(value))
            value = value.__wrapped__
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", getattr(value, "__name__", None))
        if not isinstance(module, str) or not isinstance(qualname, str):
            raise UnstableFingerprintError(
                f"Study fingerprint cannot identify callable at {path}: "
                "it has no stable module and qualified name."
            )
        if not isinstance(value, types.FunctionType):
            record = {
                "kind": "builtin_callable",
                "module": module,
                "qualname": qualname,
            }
            bound_self = getattr(value, "__self__", None)
            if bound_self is not None and not isinstance(bound_self, types.ModuleType):
                record["self"] = self.record(bound_self, path=f"{path}.__self__")
            return record

        def build() -> dict[str, Any]:
            closure = []
            for index, cell in enumerate(value.__closure__ or ()):
                try:
                    cell_value = cell.cell_contents
                except ValueError:
                    closure.append({"kind": "empty_cell"})
                else:
                    closure.append(self.record(cell_value, path=f"{path}.closure[{index}]"))
            return {
                "kind": "function",
                "module": module,
                "qualname": qualname,
                "code": self._code_record(value.__code__, path=f"{path}.__code__"),
                "defaults": self.record(value.__defaults__, path=f"{path}.__defaults__"),
                "kwdefaults": self.record(value.__kwdefaults__, path=f"{path}.__kwdefaults__"),
                "closure": closure,
            }

        return self._guarded(value, path=path, build=build)


def _solver_model_record(recorder: _Recorder, model: Any, *, path: str) -> dict[str, Any]:
    """Encode a numerical model through the solver-view contract.

    Structural identity comes from the structure signature, while the native
    theta codec supplies a deterministic flat representation of every current
    parameter value. This avoids walking authored Recipe IR or its private
    callable wrappers while still distinguishing value changes that leave the
    compile structure unchanged.
    """
    view = get_or_build_solver_view(model)
    codec = view.theta_codec()
    return {
        "structural_signature": recorder.record(
            view.structural_signature,
            path=f"{path}.structural_signature",
        ),
        "theta": recorder.record(
            codec.ravel(view.default_theta()),
            path=f"{path}.theta",
        ),
    }


def study_content_fingerprint(
    *,
    experiments: tuple,
    experiment_ids: tuple,
    parameters: Any,
    clusters: Mapping[str, tuple],
    independent: bool,
) -> str:
    """Hash the complete executable/statistical Study declaration."""
    recorder = _Recorder()
    experiment_records = []
    for index, (experiment_id, experiment) in enumerate(zip(experiment_ids, experiments)):
        where = f"experiments[{index}]"
        measurements = []
        for measurement_index, measurement in enumerate(experiment.measurements):
            measurement_path = f"{where}.measurements[{measurement_index}]"
            measurements.append(
                {
                    "name": recorder.record(measurement.name, path=f"{measurement_path}.name"),
                    "times": recorder.record(measurement.times, path=f"{measurement_path}.times"),
                    "values": recorder.record(
                        measurement.values, path=f"{measurement_path}.values"
                    ),
                    "observe": recorder.record(
                        measurement.observe, path=f"{measurement_path}.observe"
                    ),
                    "weight": recorder.record(
                        measurement.weight, path=f"{measurement_path}.weight"
                    ),
                    "likelihood": recorder.record(
                        measurement.likelihood, path=f"{measurement_path}.likelihood"
                    ),
                    "mask": recorder.record(measurement.mask, path=f"{measurement_path}.mask"),
                    "shape": recorder.record(measurement.shape, path=f"{measurement_path}.shape"),
                    "dtype": recorder.record(
                        None if measurement.dtype is None else str(measurement.dtype),
                        path=f"{measurement_path}.dtype",
                    ),
                }
            )
        record = {
            "id": recorder.record(experiment_id, path=f"{where}.id"),
            "model": {
                "content": _solver_model_record(
                    recorder,
                    experiment.model,
                    path=f"{where}.model",
                ),
                # Recipe lowering attaches these after constructing the frozen
                # dataclass. They affect the fit but are therefore absent from
                # dataclasses.fields(model) and must be folded explicitly.
                "declared_param_bounds": recorder.record(
                    getattr(experiment.model, "_declared_param_bounds", None),
                    path=f"{where}.model._declared_param_bounds",
                ),
            },
            "solver": recorder.record(experiment.solver, path=f"{where}.solver"),
            "t_span": recorder.record(experiment.t_span, path=f"{where}.t_span"),
            "initial_state": recorder.record(
                experiment.initial_state, path=f"{where}.initial_state"
            ),
            "measurements": measurements,
            "inputs": recorder.record(experiment.inputs, path=f"{where}.inputs"),
            # The trajectory partitioning selects the estimator: multiple
            # shooting adds window-start decision variables and continuity
            # rows, so two Studies differing only here are not the same
            # statistical problem and must not share provenance.
            "trajectory": recorder.record(experiment.trajectory, path=f"{where}.trajectory"),
        }
        # An operating schedule changes what is solved. The entry exists only for an
        # experiment that has one, so a study without schedules keeps its fingerprint.
        schedule = getattr(experiment, "schedule", None)
        if schedule is not None:
            record["schedule"] = recorder.record(schedule, path=f"{where}.schedule")
        # The members an experiment names in a hierarchy place it in the tree of
        # fitted values. The entry exists only for an experiment that names them.
        members = getattr(experiment, "parameters", None)
        if members is not None:
            record["parameters"] = recorder.record(members, path=f"{where}.parameters")
        experiment_records.append(record)
    declaration = {
        "contract": "diff-soup-uq-study-v2",
        "experiments": experiment_records,
        "parameters": recorder.record(parameters, path="parameters"),
        "clusters": recorder.record(clusters, path="clusters"),
        "independent": recorder.record(independent, path="independent"),
    }
    return hashlib.sha256(_canonical_json(declaration).encode("utf-8")).hexdigest()


__all__ = ["UnstableFingerprintError", "study_content_fingerprint"]
