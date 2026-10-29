"""Turn on JAX's constant hoisting, whether or not JAX is already imported.

A traced program closes over the arrays of the meshes it uses: coordinates,
connectivity, quadrature tables, sparsity colourings. By default JAX writes
them into the program text, so a lowered program's size, its compile time,
and its persistent-cache key all grow with the mesh. With
``jax_use_simplified_jaxpr_constants`` on, every such array over 32 bytes
becomes an argument of the executable instead.

JAX reads part of that setting once, when it is imported: whether a
``jax.Array`` can be a jaxpr literal, and whether the top-level executable
takes the hoisted constants as arguments (the default of
``LoweringParameters.hoist_constants_as_args``). Imported before JAX, this
module sets ``JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=1`` so JAX reads it on. When
JAX was imported first, it applies those two import-time effects itself.
Both live in JAX's private implementation, so every name is looked up first
and nothing is changed unless all of them are found; if any is missing, or
hoisting is still off afterwards, the import warns at the user's line.
"""

import os
import sys

_JAX_IMPORTED_FIRST = "jax" in sys.modules
os.environ.setdefault("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "1")

import jax  # noqa: E402

jax.config.update("jax_use_simplified_jaxpr_constants", True)

_HOIST = "hoist_constants_as_args"


def _top_level_hoisting_is_on() -> bool:
    """Whether a default lowering hoists constants and a ``jax.Array`` is a literal type."""
    from diff_soup._jax_compat import internal_array, internal_core, internal_mlir

    try:
        hoisted = internal_mlir.LoweringParameters().hoist_constants_as_args
        literal = internal_array.ArrayImpl in internal_core.literalable_types
    except Exception:  # noqa: BLE001 -- private JAX names; a missing one means "not on"
        return False
    return bool(hoisted) and literal


def _turn_on_top_level_hoisting() -> None:
    """Apply JAX's import-time hoisting effects after JAX has been imported.

    Every private name is resolved before anything changes, so a JAX whose
    internals moved is left untouched.
    """
    from diff_soup._jax_compat import internal_array, internal_core, internal_mlir

    try:
        add_literal_type = internal_core.literalable_types.add
        array_type = internal_array.ArrayImpl
        init = internal_mlir.LoweringParameters.__init__
        keyword_defaults = dict(init.__kwdefaults__ or {})
        if _HOIST in keyword_defaults:
            keyword_defaults[_HOIST] = True
            defaults = init.__defaults__
        else:
            code = init.__code__
            positional = code.co_varnames[: code.co_argcount]
            defaults = list(init.__defaults__ or ())
            index = positional.index(_HOIST) - (len(positional) - len(defaults))
            if index < 0:
                return
            defaults[index] = True
            defaults = tuple(defaults)
            keyword_defaults = init.__kwdefaults__
    except Exception:  # noqa: BLE001 -- private JAX names; leave JAX as it is
        return
    init.__defaults__ = defaults
    init.__kwdefaults__ = keyword_defaults
    add_literal_type(array_type)


def _warn_hoisting_is_off() -> None:
    """Warn at the user's import line that top-level constant hoisting is off."""
    from diff_soup._warnings import warn_user

    warn_user(
        "JAX was imported before JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS was set, and "
        "diff_soup could not turn top-level constant hoisting on afterwards: every "
        "lowered program embeds its mesh arrays, and program size, compile time, and "
        "cache keys grow with the mesh. Export JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=1 "
        "before starting Python, or import diff_soup before jax.",
        RuntimeWarning,
    )


if _JAX_IMPORTED_FIRST and not _top_level_hoisting_is_on():
    _turn_on_top_level_hoisting()
    if not _top_level_hoisting_is_on():
        _warn_hoisting_is_off()
