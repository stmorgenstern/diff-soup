"""Observed values that are loaded on the process that owns them.

A :class:`~diff_soup.inference.Measurement` may take a zero-argument callable
for its ``values``. The callable is not invoked at construction. Its shape and
dtype are either declared on the measurement or probed once on the primary
process and shared with the other processes as metadata. The values
themselves are loaded once, on the process that evaluates the experiment:
with one process and no mesh that is the lowering of the experiment; under a
multi-process mesh it is the owner of the experiment's lane, and every other
process builds the same problem over a zero placeholder of the same shape.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["DeferredValues"]

_DTYPE_CODES = tuple(
    np.dtype(name)
    for name in (
        "float32",
        "float64",
        "int32",
        "int64",
        "bool",
        "complex64",
        "complex128",
        "float16",
        "uint8",
        "int8",
        "int16",
        "uint16",
        "uint32",
        "uint64",
    )
) + (np.dtype(jnp.bfloat16),)
_MAX_NDIM = 8


class DeferredValues:
    """A measurement's observed values behind a zero-argument loader."""

    def __init__(
        self,
        load: Callable[[], Any],
        *,
        name: str,
        n_frames: int,
        shape: tuple | None = None,
        dtype: Any = None,
    ):
        """Record the loader and its declared metadata without invoking it.

        Args:
            load: Zero-argument callable returning the observed values.
            name: Measurement name, used in error messages.
            n_frames: Number of observation times the values must lead with.
            shape: Declared shape of the loaded values, or ``None`` to probe.
            dtype: Declared dtype of the loaded values, or ``None`` to probe.
        """
        if not callable(load):
            raise TypeError(f"Measurement {name!r} deferred values must be callable.")
        self.load = load
        self.name = name
        self.n_frames = int(n_frames)
        self._declared_shape = None if shape is None else tuple(int(n) for n in shape)
        self._declared_dtype = None if dtype is None else np.dtype(dtype)
        if self._declared_shape is not None:
            self._check_leading(self._declared_shape)
        self._metadata: tuple[tuple, np.dtype] | None = (
            None
            if self._declared_shape is None or self._declared_dtype is None
            else (self._declared_shape, self._declared_dtype)
        )
        self._values: jax.Array | None = None
        self.calls = 0
        self._lock = threading.Lock()

    def _check_leading(self, shape: tuple) -> None:
        if len(shape) < 1 or shape[0] != self.n_frames:
            raise ValueError(
                f"Measurement {self.name!r} values need {self.n_frames} leading frames; "
                f"got shape {tuple(shape)}."
            )

    def _load_once(self) -> jax.Array:
        with self._lock:
            if self._values is None:
                self.calls += 1
                values = jnp.asarray(self.load())
                self._check_leading(values.shape)
                if self._declared_shape is not None and tuple(values.shape) != self._declared_shape:
                    raise ValueError(
                        f"Measurement {self.name!r} declared shape {self._declared_shape} but "
                        f"its values loaded with shape {tuple(values.shape)}."
                    )
                if self._declared_dtype is not None and values.dtype != self._declared_dtype:
                    raise ValueError(
                        f"Measurement {self.name!r} declared dtype {self._declared_dtype} but "
                        f"its values loaded with dtype {values.dtype}."
                    )
                self._values = values
                self._metadata = (tuple(values.shape), np.dtype(values.dtype))
            return self._values

    @property
    def loaded(self) -> bool:
        """Whether the values have been loaded on this process."""
        return self._values is not None

    def metadata(self) -> tuple[tuple, np.dtype]:
        """Return ``(shape, dtype)`` without loading the values on a non-primary process.

        Declared metadata is returned as is. Otherwise the primary process loads
        the values once and every process receives the shape and dtype through a
        collective, so each process calls this at the same point of its program.
        """
        if self._metadata is not None:
            return self._metadata
        if jax.process_count() == 1:
            self._load_once()
            return self._metadata
        from diff_soup._jax_compat import multihost_utils

        record = np.zeros((_MAX_NDIM + 2,), dtype=np.int64)
        if jax.process_index() == 0:
            self._load_once()
            shape, dtype = self._metadata
            if len(shape) > _MAX_NDIM:
                raise ValueError(
                    f"Measurement {self.name!r} values have {len(shape)} dimensions; "
                    f"deferred values support at most {_MAX_NDIM}."
                )
            record[0], record[1] = len(shape), _dtype_code(dtype)
            record[2 : 2 + len(shape)] = shape
        record = np.asarray(multihost_utils.broadcast_one_to_all(record))
        ndim, code = int(record[0]), int(record[1])
        self._metadata = (tuple(int(n) for n in record[2 : 2 + ndim]), _DTYPE_CODES[code])
        return self._metadata

    def materialize(self) -> jax.Array:
        """Load the values on this process, once, and return them."""
        return self._load_once()

    def placeholder(self) -> jax.Array:
        """Zeros of the values' shape and dtype, for a process that does not own them."""
        shape, dtype = self.metadata()
        return jnp.zeros(shape, dtype=dtype)


def _dtype_code(dtype: np.dtype) -> int:
    for code, candidate in enumerate(_DTYPE_CODES):
        if candidate == dtype:
            return code
    raise TypeError(
        f"Deferred measurement values of dtype {dtype} cannot be shared across processes; "
        "declare a floating, integer, boolean, or complex dtype."
    )
