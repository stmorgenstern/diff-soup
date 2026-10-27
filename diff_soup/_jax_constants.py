"""Turn on JAX's constant hoisting before JAX is imported.

A traced program closes over the arrays of the meshes it uses: coordinates,
connectivity, quadrature tables, sparsity colourings. By default JAX writes
them into the program text, so a lowered program's size, its compile time,
and its persistent-cache key all grow with the mesh. With
``jax_use_simplified_jaxpr_constants`` on, every such array over 32 bytes
becomes an argument of the executable instead.

JAX fixes part of that behaviour (whether the top-level executable takes the
hoisted arguments) when it is imported, so the setting has to be in place
before ``import jax``: either import ``diff_soup`` before ``jax``, or export
``JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=1`` before starting Python. The
configuration update below applies the remaining part of the behaviour
(deduplicated, argument-passed constants inside the program) even when JAX
was imported first.
"""

import os
import sys
import warnings

_JAX_IMPORTED_FIRST = "jax" in sys.modules
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")

import jax  # noqa: E402

jax.config.update("jax_use_simplified_jaxpr_constants", True)


def _top_level_hoisting_is_on() -> bool:
    """Whether JAX bound constant hoisting into its lowering defaults at import."""
    try:
        import dataclasses

        from diff_soup._jax_compat import internal_mlir

        field = next(
            field
            for field in dataclasses.fields(internal_mlir.LoweringParameters)
            if field.name == "hoist_constants_as_args"
        )
    except Exception:  # noqa: BLE001 -- a JAX without the field hoists nothing to check
        return True
    return bool(field.default)


if _JAX_IMPORTED_FIRST and not _top_level_hoisting_is_on():
    warnings.warn(
        "JAX was imported before JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS was set, so top-level "
        "constant hoisting is off for this process: every lowered program embeds its mesh "
        "arrays, and program size, compile time, and cache keys grow with the mesh. Export "
        "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=1 before starting Python, or import diff_soup "
        "before jax.",
        RuntimeWarning,
        stacklevel=2,
    )
