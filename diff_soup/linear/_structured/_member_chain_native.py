"""The threaded native block Thomas kernel: the processor path of ``MemberSchur(elimination="native")``.

Same signature, factor ([SweepFactor][]) and singular rule as the plain-JAX sweep in ``_member_chain.py``: arrays
``(M, N, b, b)`` of many independent block-tridiagonal chains (members), right-hand sides ``(M, N, b, m)``. Block row ``i`` of a chain is
``lower[i] x[i-1] + diag[i] x[i] + upper[i] x[i+1]``; ``lower[:, 0]`` and ``upper[:, N-1]`` are never read.

The work is one XLA custom call per factor or solve (``jax.ffi.ffi_call``, not a host callback), which loops over
members under OpenMP; the thread count follows ``OMP_NUM_THREADS``. Any axis before the last three is a member axis,
so a lane axis added by ``vmap`` folds into the member loop. The calls have no derivative rule: ``ds.linear``
differentiates its phased primitives through the solves (``_phased_ad.py``), never through a factorization or the
lowering that runs this kernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from diff_soup._errors import ConfigurationError
from diff_soup._internal import member_chain_ffi
from diff_soup.linear._structured._member_chain import SweepFactor

Array = jax.Array

__all__ = ["available", "chain_factor", "chain_solve"]

_PREFIX = "diffsoup_member_chain"


def available() -> bool:
    """Whether the native module is built in this checkout."""
    return member_chain_ffi.available()


def _suffix(dtype) -> str:
    """The target suffix for ``dtype``, loading the module (which registers the targets) first."""
    try:
        member_chain_ffi.load()
    except ImportError as exc:
        raise ConfigurationError(
            'MemberSchur(elimination="native") needs the native member chain module. '
            + str(exc)
            + 'Or use elimination="thomas", the plain-JAX sweep.'
        ) from exc
    dtype = np.dtype(dtype)
    if dtype == np.float64:
        return "f64"
    if dtype == np.complex128:
        return "c128"
    raise ConfigurationError(
        f'MemberSchur(elimination="native") takes float64 or complex128 values; got {dtype}.'
    )


def chain_factor(lower: Array, diag: Array, upper: Array) -> tuple[SweepFactor, Array]:
    """Block Thomas factor of a batch of block-tridiagonal chains.

    Args:
        lower: Sub-diagonal blocks, shape ``(M, N, b, b)``.
        diag: Diagonal blocks, the same shape.
        upper: Super-diagonal blocks, the same shape.

    Returns:
        ``(factor, bad)``: a [SweepFactor][] holding the inverse pivot blocks ``inv``, ``cp = inv @ upper`` and
        ``low``, ``lower`` with its first site zeroed, each ``(M, N, b, b)``; and a boolean ``(M,)`` that holds
        where a member has a negligible or non-finite pivot, by the plain-JAX sweep's rule.
    """
    dtype = jnp.result_type(lower, diag, upper)
    suffix = _suffix(dtype)
    lower, diag, upper = (jnp.asarray(a, dtype) for a in (lower, diag, upper))
    block = jax.ShapeDtypeStruct(diag.shape, dtype)
    bad = jax.ShapeDtypeStruct(diag.shape[:-3], jnp.bool_)
    inv, cp, bad = jax.ffi.ffi_call(
        f"{_PREFIX}_factor_{suffix}", (block, block, bad), vmap_method="broadcast_all"
    )(lower, diag, upper)
    return SweepFactor(inv=inv, cp=cp, low=lower.at[..., 0, :, :].set(0)), bad


def chain_solve(factor: SweepFactor, rhs: Array, transpose: bool = False) -> Array:
    """Solve with a [chain_factor][] result.

    Args:
        factor: The [SweepFactor][] from [chain_factor][].
        rhs: Right-hand sides, shape ``(M, N, b, m)``.
        transpose: Solve with the plain (never conjugated) transpose of each chain.

    Returns:
        ``x`` with the shape of ``rhs``.
    """
    inv, cp, low = factor.inv, factor.cp, factor.low
    dtype = jnp.result_type(inv, rhs)
    suffix = _suffix(dtype)
    inv, cp, low, rhs = (jnp.asarray(a, dtype) for a in (inv, cp, low, rhs))
    return jax.ffi.ffi_call(
        f"{_PREFIX}_solve_{suffix}", jax.ShapeDtypeStruct(rhs.shape, dtype), vmap_method="broadcast_all"
    )(inv, cp, low, rhs, transpose=bool(transpose))
