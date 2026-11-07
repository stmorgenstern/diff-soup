"""How the phased primitives run: eagerly, at the JAX level, or through a backend's lowering hook, at a keyed site."""

from __future__ import annotations

import functools
import hashlib
import itertools
from collections.abc import Callable, Sequence
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._jax_compat import (
    Primitive,
    ir,
    mlir,
)
from diff_soup.linear._factor import FactorContractError
from diff_soup.linear._phased import _FUNS, _HOOKS, _backend_label
from diff_soup.linear._phased_abstract import _is_exact, _solve_flags, _split_solve_operands


def storage_key(storage: Optional[str]) -> np.int64:
    """The 64-bit key a native handler receives for a shared-storage name; ``0`` for none.

    A stable hash of the name (the same in every process), so a compiled
    program that names the storage has the same text whenever it is lowered.

    Args:
        storage: The storage name, or ``None``.

    Returns:
        A positive ``int64`` for a name, ``0`` for ``None``.
    """
    if storage is None:
        return np.int64(0)
    digest = hashlib.blake2b(storage.encode("utf-8"), digest_size=8).digest()
    return np.int64((int.from_bytes(digest, "little") & ((1 << 62) - 1)) | 1)


#: The entry of a module's lowering cache that counts the contract calls lowered into it.
_SITE_ORDINALS = object()


def _lowered_device(ctx: Any, kind: str, per_device: bool) -> tuple:
    """The platform and id of the device a native call runs on, as key text.

    A site's slots are keyed by context and lane, and a cuDSS slot lives on
    the device that created it, so a site belongs to one device. JAX keys its
    trace on ``jax.default_device`` and lowers one program text for any
    single device, so the device is the default device, or, in a
    ``shard_map`` body, the one device of this process on the mesh. A cuDSS
    slot used from another device (inputs committed elsewhere) refuses the
    call. This process's first device keeps the plain key, so a program on it
    reads as it always did. A call whose slot the extension keys by the device
    that runs it (``per_device``, shared storage) may run in a ``shard_map``
    body over several devices.

    Raises:
        FactorContractError: If the program runs the call on several devices
            of this process, which would share the site's native storage.
    """
    axis_context = ctx.module_context.axis_context
    mesh = getattr(axis_context, "mesh", None)
    if mesh is not None:
        here = jax.process_index()
        devices = [device for device in mesh.devices.flat if device.process_index == here]
        device = devices[0] if len(devices) == 1 else None
    else:
        devices = range(int(getattr(axis_context, "num_devices", 1)))
        device = jax.config.jax_default_device
    if len(devices) > 1 and not (per_device and mesh is not None):
        raise FactorContractError(
            f"A native {kind} call is lowered into a program over {len(devices)} devices of this "
            "process, and its factor storage does not tell devices apart. Run the call on one "
            "device per process, or factor with diff_soup.linear.Dense, whose factors are arrays."
        )
    if isinstance(device, str):
        device = jax.local_devices(backend=device)[0]
    if device is None or device == jax.local_devices()[0]:
        return ()
    return (f"device {device.platform} {device.id}",)


def lowered_site_key(ctx: Any, kind: str, *parts: Any, per_device: bool = False) -> np.int64:
    """A deterministic key of one lowered native factor or refactor call, for its ``site``.

    The key is a stable 64-bit hash of the module's symbol name, the call's
    name stack, the ordinal of this call among the calls this function has
    keyed in the module so far, ``kind`` and ``parts`` (the analysis id,
    orientation, precision, lane count and whatever else tells the call
    apart). Two lowerings of one program under one default device therefore
    give the same keys and the same program text (so JAX's persistent compile
    cache recognises the program), and the native handlers create a site on the
    first call that carries an unknown key; the calls of one program, and of
    distinct programs, get distinct keys, as do lowerings for different
    devices, told apart by platform and id
    ([_lowered_device][diff_soup.linear._phased_impl._lowered_device]).
    The key has bit 62 set, so it never equals an id the extensions number their
    sites with by themselves.

    Args:
        ctx: The lowering rule context.
        kind: What the call does (``"factor"``, ``"refactor"``).
        *parts: Further values that identify the call, turned into text.
        per_device: Whether the extension keys the call's slot by the device
            that runs it (a call on shared storage).

    Returns:
        The key, a positive ``int64``.

    Raises:
        FactorContractError: If the program runs the call on several devices
            of this process and ``per_device`` does not hold in a ``shard_map``
            body.
    """
    device = _lowered_device(ctx, kind, per_device)
    module = ctx.module_context
    counters = module.cached_primitive_lowerings
    ordinal = counters.get(_SITE_ORDINALS, 0)
    counters[_SITE_ORDINALS] = ordinal + 1
    try:
        name = ir.StringAttr(module.module.operation.attributes["sym_name"]).value
    except Exception:  # noqa: BLE001 - a module without a name keys on the rest
        name = ""
    text = "|".join(str(part) for part in (name, ctx.name_stack, ordinal, kind, *parts, *device))
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return np.int64((int.from_bytes(digest, "little") & ((1 << 62) - 1)) | (1 << 62))


# --------------------------------------------------------------------------- #
# Evaluation and lowering
# --------------------------------------------------------------------------- #
def _lowering_hook(kind: str, params: dict) -> Optional[Callable]:
    """The backend's MLIR lowering hook for ``kind``, or ``None``."""
    return getattr(params["analysis"].backend, _HOOKS[kind], None)


def _conform(kind: str, analysis: Any, outs: Any, avals_out: Sequence[Any]) -> list:
    """Check a JAX-level backend function's outputs against the abstract outputs."""
    outs = list(outs)
    label, fun = _backend_label(analysis), _FUNS[kind]
    if len(outs) != len(avals_out):
        raise FactorContractError(
            f"The {label} backend's {fun} returned {len(outs)} outputs; the contract requires "
            f"{len(avals_out)}, the order word last."
        )
    conformed = []
    for position, (out, aval) in enumerate(zip(outs, avals_out)):
        out = jnp.asarray(out)
        if tuple(out.shape) != tuple(aval.shape):
            raise FactorContractError(
                f"The {label} backend's {fun} returned output {position} with shape "
                f"{list(out.shape)}; the contract requires {list(aval.shape)}."
            )
        conformed.append(out if out.dtype == aval.dtype else out.astype(aval.dtype))
    return conformed


def _jax_level(kind: str, avals_out: Sequence[Any]) -> Callable:
    """The backend's JAX-level function for ``kind``, called with the primitive's operands.

    Each function receives the order word as the keyword ``order`` and
    returns it as its last output. An invalidation has no JAX-level form:
    it returns the order word unchanged.
    """

    def run(*operands: Any, **params: Any) -> list:
        analysis = params["analysis"]
        if kind == "invalidate":
            # Without a lowering hook there is no native storage to mark.
            return _conform(kind, analysis, (operands[-1],), avals_out)
        backend = analysis.backend
        common = dict(
            analysis=analysis,
            orientation=params["orientation"],
            precision=params["precision"],
            tol=params["tol"],
        )
        if kind == "factor":
            values, token, context_id, order = operands
            outs = backend.factor_fun(values, token, context_id, order=order, **common)
        elif kind == "apply":
            values, *rest = operands
            handle, order, b, weights, x0 = _split_solve_operands(
                "phased_solve", rest, params["n_handle"], *_solve_flags(params)
            )
            if not _is_exact(analysis):
                # An inexact backend also receives the caller's weights and initial guess.
                common.update(weights=weights, x0=x0)
            outs = backend.apply_fun(
                values, *handle, order=order, b=b, transpose=params["transpose"], **common
            )
        else:
            values, held_values, predicate, *handle, order = operands
            outs = backend.refactor_fun(
                values,
                held_values,
                predicate,
                *handle,
                order=order,
                adopt=params["adopt"],
                **common,
            )
        return _conform(kind, analysis, outs, avals_out)

    return run


@functools.lru_cache(maxsize=64)
def _compiled(primitive: Primitive, frozen_params: tuple) -> Callable:
    """A compiled program that binds ``primitive`` once with fixed parameters."""
    params = dict(frozen_params)

    def run(*operands: Any) -> Any:
        return primitive.bind(*operands, **params)

    return jax.jit(run)


#: Slot tokens of eager factorizations through a lowering hook: each eager
#: call takes the next one, so no two eager factorizations of the process
#: share a (context, call site, slot token) key and each keeps storage of its
#: own.
_EAGER_TOKENS = itertools.count(1)


def _eager_token() -> jax.Array:
    """A slot token no earlier eager factorization of this process has used (wrapping at ``2**31``)."""
    return jnp.asarray(next(_EAGER_TOKENS) % (2**31), jnp.int32)


def _impl(kind: str, primitive: Primitive) -> Callable:
    """Eager evaluation: the JAX-level function, or a compiled bind for a lowering hook.

    A backend with a lowering hook runs a compiled one-equation program,
    cached per parameter set, so an eager call takes the same native route
    as a call inside ``jit``. Every eager factorization through it gets a
    fresh slot token, so eager factors never share a slot of that program's
    call site. A backend without one receives every operand as a JAX
    array: an operand bound as a NumPy scalar (the context id and slot
    token [factor][diff_soup.linear.factor] binds, so that they lower as
    program constants) is converted first.
    """

    def impl(*operands: Any, **params: Any) -> Any:
        if _lowering_hook(kind, params) is not None:
            if kind == "factor":
                values, _, context_id, order = operands
                operands = (values, _eager_token(), context_id, order)
            with jax.disable_jit(False):
                return _compiled(primitive, tuple(sorted(params.items())))(*operands)
        operands = tuple(
            jnp.asarray(op) if isinstance(op, (np.ndarray, np.generic)) else op for op in operands
        )
        avals_out = primitive.abstract_eval(*(jax.typeof(op) for op in operands), **params)[0]
        return _jax_level(kind, avals_out)(*operands, **params)

    return impl


def _check_platform(ctx: Any, analysis: Any) -> None:
    """Refuse a lowering for a platform outside the backend's ``platforms``.

    Raises:
        FactorContractError: Naming the backend, the platform and device the
            program is compiled for, and the platforms the backend runs on.
    """
    allowed = tuple(getattr(analysis.backend, "platforms", ()) or ())
    if not allowed:
        return
    lowered_for = tuple(getattr(ctx.module_context, "platforms", ()) or ())
    outside = [platform for platform in lowered_for if platform not in allowed]
    if not outside:
        return
    try:
        device = str(jax.devices(outside[0])[0])
    except RuntimeError:
        device = "unavailable"
    raise FactorContractError(
        f"The {_backend_label(analysis)} backend runs on the {' or '.join(allowed)} platform "
        f"only, but this program is compiled for the {' and '.join(outside)} platform "
        f"(device {device}). Compile it for {allowed[0]}: give it {allowed[0]} arrays or run it "
        f"under jax.default_device(jax.devices({allowed[0]!r})[0]), or analyze with a backend "
        f"that runs on {outside[0]}."
    )


def _lowering(kind: str) -> Callable:
    """The lowering rule for ``kind``: the backend's hook, else ``mlir.lower_fun``.

    The rule runs once per equation (the primitives are registered
    uncacheable). A hook receives every operand,
    the order word included, and the ``storage`` parameter only when it
    names shared storage, and returns the MLIR values of every result, the
    order word last. A program lowered for a platform the backend does not
    declare in ``platforms`` is refused here, before any custom call is
    emitted.
    """

    def rule(ctx: Any, *args: Any, **params: Any) -> Any:
        _check_platform(ctx, params["analysis"])
        hook = _lowering_hook(kind, params)
        if hook is not None:
            if params.get("storage", "") is None:
                # A hook receives the storage name only for a factor that has one.
                params = {key: value for key, value in params.items() if key != "storage"}
            return hook(ctx, *args, **params)
        lowered = mlir.lower_fun(_jax_level(kind, ctx.avals_out), multiple_results=True)
        return lowered(ctx, *args, **params)

    return rule
