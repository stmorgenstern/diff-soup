"""Memory checks before a dense Hessian or residual Jacobian is formed, and the tiled Jacobian."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError

# The memory of an exact Hessian grows with the square of the parameter count:
# every tangent direction of the forward-over-reverse product carries the whole
# reverse tape. Measured on a decision-variable multishoot fit (CPU,
# 2026-10-28): 4.2 / 16.1 / 63.2 GB of compiled temporaries at 83 / 163 / 323
# parameters. Below this width the Hessian is used without measuring it first.
_HESSIAN_GUARD_MIN_WIDTH = 64


#: The share of the free device memory one Hessian evaluation may take.
_HESSIAN_MEMORY_SHARE = 0.9


class _HessianMemoryError(ConfigurationError):
    """The exact Hessian needs more memory than the device has free."""


def _cgroup_memory_headroom() -> Optional[int]:
    """The bytes this process may still allocate under its cgroup v2 limits, or ``None``."""
    try:
        with open("/proc/self/cgroup") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    relative = next((line[3:] for line in lines if line.startswith("0::")), None)
    if relative is None:
        return None
    headroom = None
    path = os.path.normpath("/sys/fs/cgroup/" + relative.lstrip("/"))
    while path.startswith("/sys/fs/cgroup"):
        try:
            with open(os.path.join(path, "memory.max")) as handle:
                limit = handle.read().strip()
            with open(os.path.join(path, "memory.current")) as handle:
                current = int(handle.read().strip())
        except (OSError, ValueError):
            limit = "max"
            current = 0
        if limit != "max":
            room = max(int(limit) - current, 0)
            headroom = room if headroom is None else min(headroom, room)
        if path == "/sys/fs/cgroup":
            break
        path = os.path.dirname(path)
    return headroom


def _host_available_memory() -> Optional[int]:
    """The host memory available to this process in bytes, or ``None`` when unknown."""
    available = None
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        available = None
    if available is None:
        try:
            import psutil

            available = int(psutil.virtual_memory().available)
        except Exception:  # psutil is optional; without it the host is unmeasured
            available = None
    headroom = _cgroup_memory_headroom()
    if headroom is not None:
        available = headroom if available is None else min(available, headroom)
    return available


def _free_device_memory(device: Any) -> Optional[int]:
    """The bytes ``device`` can still allocate, or ``None`` when it cannot say."""
    if device.platform == "cpu":
        return _host_available_memory()
    try:
        stats = device.memory_stats() or {}
    except Exception:  # a backend without allocator statistics is unmeasured
        return None
    limit = stats.get("bytes_limit")
    if limit is None:
        return None
    return int(limit) - int(stats.get("bytes_in_use", 0))


def _memory_need_and_free(compiled: Any) -> Optional[tuple]:
    """``(need, free, platform)`` of a compiled evaluation, or ``None`` when unmeasured."""
    try:
        analysis = compiled.memory_analysis()
    except Exception:  # not every backend implements the analysis
        return None
    if analysis is None:
        return None
    need = int(getattr(analysis, "temp_size_in_bytes", 0) or 0) + int(
        getattr(analysis, "output_size_in_bytes", 0) or 0
    )
    device = jax.local_devices()[0]
    free = _free_device_memory(device)
    if free is None:
        return None
    return need, free, device.platform


def _check_hessian_memory(compiled: Any, width: int, tile: Any) -> None:
    """Raise `_HessianMemoryError` when a compiled Hessian needs more than the free memory.

    ``compiled`` is the ahead-of-time compiled evaluation of ``width``
    parameters in tiles of ``tile`` directions. A backend that reports no
    memory figures passes.
    """
    measured = _memory_need_and_free(compiled)
    if measured is None or measured[0] <= _HESSIAN_MEMORY_SHARE * measured[1]:
        return
    need, free, platform = measured
    gib = 2.0**30
    how = "whole" if tile in (None, "full") else f"in tiles of {tile} directions"
    smaller = (
        "a smaller hessian_tile= (8 or 16), " if tile in (None, "full") or int(tile) > 8 else ""
    )
    raise _HessianMemoryError(
        f"The exact Hessian over {width} parameters, formed {how}, needs "
        f"{need / gib:.1f} GiB and the {platform} device has {free / gib:.1f} GiB "
        f"free. With NewtonTrustRegion pass {smaller}"
        'hessian="gauss-newton", or subproblem="cg" with '
        "x_scale=ds.optimization.inference.ConstantScale(...). For laplace and information, "
        'use the expected curvature (curvature="expected", kind="expected").'
    )


class _JacobianMemoryError(ConfigurationError):
    """The Gauss-Newton residual Jacobian needs more memory than the device has free."""


def _jacobian_memory_reporter(knob: str) -> Callable:
    """The memory check of a residual Jacobian whose tile the option ``knob`` sets."""

    def check(compiled: Any, width: int, tile: Any) -> None:
        measured = _memory_need_and_free(compiled)
        if measured is None or measured[0] <= _HESSIAN_MEMORY_SHARE * measured[1]:
            return
        need, free, platform = measured
        gib = 2.0**30
        how = "whole" if tile in (None, "full") else f"in tiles of {tile} directions"
        smaller = (
            f"Pass a smaller {knob}= (8 or 16), or d"
            if tile in (None, "full") or int(tile) > 8
            else "D"
        )
        raise _JacobianMemoryError(
            f"The Gauss-Newton residual Jacobian over {width} parameters, formed {how}, "
            f"needs {need / gib:.1f} GiB and the {platform} device has {free / gib:.1f} GiB "
            f"free. {smaller}eclare a ds.optimization.inference.StateBasis or fewer windows to shrink "
            "the decision vector."
        )

    return check


#: Set while a concrete check compiles: an enclosing check measures every
#: nested Hessian, so the nested ones skip their own compilation.
_ENCLOSING_CHECK = threading.local()


class _MemoryCheckedHessian:
    """A jitted Hessian evaluation whose memory is checked before its first run.

    Above `_HESSIAN_GUARD_MIN_WIDTH` parameters the evaluation is compiled
    ahead of time once per input shape and its memory compared with the free
    memory of the device. A concrete input is compiled exactly as the call
    compiles it, so the call reuses that compilation. A traced input (the
    Hessian inside another jitted function) is measured on its shape, which
    costs a second compilation, unless an enclosing check is compiling it.
    ``report(compiled, width, tile)`` raises when the evaluation does not fit;
    the exact Hessian's check by default, a residual Jacobian's through
    `_jacobian_memory_reporter`.
    """

    def __init__(self, fn: Callable, *, tile: Any = "full", report: Callable = None):
        self._fn = fn if hasattr(fn, "lower") else jax.jit(fn)
        self.tile = tile
        self._report = _check_hessian_memory if report is None else report
        self._checked: set = set()

    def lower(self, *args, **kwargs):
        """Lower the wrapped jitted function, as `jax.jit(...).lower` does."""
        return self._fn.lower(*args, **kwargs)

    def check(self, x: Any) -> None:
        """Compile for ``x`` and raise the memory error of the evaluation when it does not fit."""
        shape = tuple(np.shape(x))
        if len(shape) != 1 or shape[0] <= _HESSIAN_GUARD_MIN_WIDTH:
            return
        key = (shape, str(jnp.result_type(x)))
        traced = isinstance(x, jax.core.Tracer)
        if key in self._checked or (traced and getattr(_ENCLOSING_CHECK, "active", False)):
            return
        point = jax.ShapeDtypeStruct(shape, jnp.result_type(x)) if traced else x
        enclosing = getattr(_ENCLOSING_CHECK, "active", False)
        _ENCLOSING_CHECK.active = True
        try:
            compiled = self._fn.lower(point).compile()
        finally:
            _ENCLOSING_CHECK.active = enclosing
        self._report(compiled, shape[0], self.tile)
        self._checked.add(key)

    def __call__(self, x: Any):
        self.check(x)
        return self._fn(x)


#: The tiles ``"auto"`` tries for a residual Jacobian that does not fit whole, widest first.
_AUTO_JACOBIAN_TILES = (32, 16, 8)


class _TiledJacobian:
    """A jitted ``x -> (residual, jacobian)`` that can also be formed in tiles.

    ``build(tile)`` returns the evaluation pushing ``tile`` tangents at a time;
    calling the object evaluates it with every tangent at once. The solvers
    read [at_tile][diff_soup.optimization.inference.fitting._residual_memory._TiledJacobian.at_tile]
    to tile a structured Jacobian instead of tiling the residual's directions.
    """

    def __init__(self, build: Callable):
        self._build = build
        self._jitted: dict = {}

    def at_tile(self, tile: Any) -> Callable:
        """The jitted evaluation in tiles of ``tile`` tangents, built once per tile."""
        key = "full" if tile is None else tile
        if key not in self._jitted:
            self._jitted[key] = jax.jit(self._build(key))
        return self._jitted[key]

    def lower(self, x: Any):
        """Lower the whole evaluation, as `jax.jit(...).lower` does."""
        return self.at_tile("full").lower(x)

    def __call__(self, x: Any):
        return self.at_tile("full")(x)


def checked_residual_jacobian(problem: Any, x: Any, tile: Any, *, knob: str) -> Any:
    """``x -> (residual, jacobian)`` of a least-squares problem, memory-checked at ``x``.

    The problem's structured Jacobian (a `_TiledJacobian`, such as fitted
    window starts give) is used when it has one, else the residual's forward
    Jacobian. ``tile`` is an int, ``"full"``, or ``"auto"``: whole when the
    compiled evaluation fits in the free device memory, else the widest of
    `_AUTO_JACOBIAN_TILES` that fits. The result's ``tile`` names the tile
    used. ``knob`` is the solver option the memory error names.

    Raises:
        ConfigurationError: If no candidate tile fits.
    """
    from diff_soup.optimization.solvers.tangent import value_and_jacfwd_tiled

    at_tile = getattr(problem.residual_and_jacobian, "at_tile", None)
    residual = problem.residual

    def evaluation(chosen):
        if at_tile is not None:
            return at_tile(chosen)
        return jax.jit(lambda point: value_and_jacfwd_tiled(residual, point, tile=chosen))

    width = int(np.shape(x)[0])
    if tile == "auto":
        candidates = ["full"] + [t for t in _AUTO_JACOBIAN_TILES if t < width]
    else:
        candidates = ["full" if tile is None else tile]
    report = _jacobian_memory_reporter(knob)
    error = None
    for candidate in candidates:
        checked = _MemoryCheckedHessian(evaluation(candidate), tile=candidate, report=report)
        try:
            checked.check(x)
        except _JacobianMemoryError as exc:
            error = exc
            continue
        return checked
    raise error
