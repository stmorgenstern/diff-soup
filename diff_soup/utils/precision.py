"""Float32 / float64 precision guard for diff_soup solvers.

diff_soup defaults to whatever JAX's ``jax_enable_x64`` flag says (float32 by
default). For stiff PDE/DAE problems float32 can silently compromise accuracy
— mass-conservation errors around 1e-4, non-convergent Newton iterations, and
discretisation errors that dominate the physics.

This module emits a **one-time warning** the first time any solver entry point
is called with x64 disabled, pointing users at the fix. It does NOT auto-enable
x64; precision remains a user decision.
"""

from __future__ import annotations

import jax

from diff_soup._warnings import warn_user

__all__ = ["warn_if_float32_default", "warn_if_float32_on_cpu"]

_WARNED: bool = False
_CPU_FLOAT32_WARNED: bool = False

_CPU_FLOAT32_MSG = (
    "a float32 problem is being solved on a CPU device. float32 is not "
    "recommended on CPU: a CPU gains little from single precision, and the "
    "solve gives up float64 accuracy for it. Build the problem with a float64 "
    "Precision for CPU solves; keep float32 for GPU solves. This warning fires "
    "once per process."
)

_WARN_MSG = (
    "diff_soup solver called with jax_enable_x64=False (float32 default). "
    "Stiff PDE/DAE systems frequently need float64 for accurate mass "
    "conservation, stable Newton convergence, and convergence-order-"
    "respecting discretisation.  To enable float64:\n"
    "    import jax\n"
    '    jax.config.update("jax_enable_x64", True)\n'
    "This warning fires once per process."
)


def warn_if_float32_default(stacklevel: int = 3) -> None:
    """Emit a one-shot warning if JAX is running in float32 (x64 disabled) mode.

    Call this from every public solver entry point before the solve begins.
    The function is idempotent: it fires at most once per process regardless
    of how many solver calls are made.

    Args:
        stacklevel: Ignored. The warning is attributed to the first frame
            outside diff_soup, equinox, jax and jaxtyping.
    """
    global _WARNED
    if _WARNED:
        return
    try:
        x64_on = bool(jax.config.read("jax_enable_x64"))
    except Exception:
        x64_on = False
    if not x64_on:
        warn_user(_WARN_MSG, UserWarning)
    _WARNED = True


def warn_if_float32_on_cpu(precision, *sample_arrays, stacklevel: int = 3) -> None:
    """Emit a one-shot warning when a float32 problem is solved on a non-GPU device.

    The solve itself is unchanged: a backend that serves float32 still runs it
    in float32, and one that does not still refuses it.

    Args:
        precision: The resolved [Precision][diff_soup._precision.Precision] of
            the solve.
        *sample_arrays: The solve's inputs (initial state, parameters, time
            span); the first placed array names the device.
        stacklevel: Ignored, as for
            [warn_if_float32_default][diff_soup.utils.precision.warn_if_float32_default].
    """
    global _CPU_FLOAT32_WARNED
    if _CPU_FLOAT32_WARNED:
        return
    import numpy as np

    if np.dtype(precision.compute_dtype).itemsize > 4:
        return
    from diff_soup.utils.placement import array_platform

    if array_platform(*sample_arrays) == "gpu":
        return
    _CPU_FLOAT32_WARNED = True
    warn_user(_CPU_FLOAT32_MSG, UserWarning)
