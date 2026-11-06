"""The execution platform of a solve's inputs.

A solve reads its platform (CPU or GPU) off the arrays it is handed rather
than off a process-wide default, so a program placed on one device keeps
that placement under ``vmap`` and ``jit``. The linear-backend resolver and
the precision policy both decide platform-dependent behaviour from this.
"""

from __future__ import annotations

import jax

__all__ = ["array_platform", "default_platform", "is_batch_tracer"]


def is_batch_tracer(value) -> bool:
    """Whether ``value`` is the tracer ``vmap`` wraps around a batched array.

    JAX does not export its batch-tracer class, so the check is structural: a
    tracer carrying the batched value (``val``) and the batch position
    (``batch_dim``) is a ``vmap`` tracer; forward-mode and staging tracers
    carry neither.
    """
    return (
        isinstance(value, jax.core.Tracer) and hasattr(value, "val") and hasattr(value, "batch_dim")
    )


def default_platform() -> str:
    """Return the platform a program with uncommitted inputs runs on.

    This is the device of the enclosing ``jax.default_device``, which holds per
    thread, and otherwise the process's default backend. A program whose inputs
    are committed to a device runs there whatever this says, so under ``jit``
    such a lane names its linear backend or runs under ``jax.default_device``.
    """
    device = jax.config.jax_default_device
    platform = jax.default_backend() if device is None else getattr(device, "platform", device)
    return "gpu" if platform in ("gpu", "cuda", "rocm") else "cpu"


def array_platform(*sample_arrs) -> str:
    """Return the execution platform of the first non-empty sample array.

    A direct ``vmap`` wraps an already placed array in one or more
    ``BatchTracer`` objects. Unwrapping their value channel preserves the
    actual CPU/GPU placement instead of misclassifying every batched input as
    CPU. A staging tracer created by an enclosing ``jit`` has no concrete
    device, and the fallback is [default_platform][diff_soup.utils.placement.default_platform].
    """
    for sample in sample_arrs:
        for leaf in jax.tree_util.tree_leaves(sample):
            a = leaf
            depth = 0
            while is_batch_tracer(a) and depth < 8:
                a = a.val
                depth += 1
            if a is None or int(getattr(a, "size", 0)) == 0:
                continue
            try:
                return next(iter(a.devices())).platform
            except Exception:  # noqa: BLE001 — a staging tracer has no device
                return default_platform()
    return default_platform()
