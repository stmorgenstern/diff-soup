"""The threaded native block Thomas kernel: the chain kernel of ``MemberSchur(elimination="native")``.

Same factor ([SweepFactor][]) and singular rule as the plain-JAX sweep in ``_member_chain.py``, on the same flat
chain: blocks ``(S, b, b)`` of members stored back to back, right-hand sides ``(S, b, m)``, and the members'
``offsets`` ``(M + 1,)``. Each member is swept at its own length.

The work is one XLA custom call per factor or solve (``jax.ffi.ffi_call``, not a host callback), which loops over
members under OpenMP when there is enough work to share; the thread count follows ``OMP_NUM_THREADS``. A lane axis
added by ``vmap`` folds into the member loop. The calls have no derivative rule: ``ds.linear`` differentiates its
phased primitives through the solves (``_phased_ad.py``), never through a factorization or the lowering that runs
this kernel.
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


def chain_factor(lower: Array, diag: Array, upper: Array, offsets: np.ndarray) -> tuple[SweepFactor, Array]:
    """Block Thomas factor of the members of a flat chain.

    Args:
        lower: Sub-diagonal blocks, shape ``(S, b, b)``, zero at a member's first site.
        diag: Diagonal blocks, the same shape.
        upper: Super-diagonal blocks, the same shape.
        offsets: ``(M + 1,)`` ``int32`` first site of every member, and ``S``.

    Returns:
        ``(factor, bad)``: a [SweepFactor][] holding the inverse pivot blocks ``inv``, ``cp = inv @ upper`` and
        ``low = lower``, each ``(S, b, b)``; and one boolean that holds where some member has a negligible or
        non-finite pivot, by the plain-JAX sweep's rule.
    """
    dtype = jnp.result_type(lower, diag, upper)
    suffix = _suffix(dtype)
    lower, diag, upper = (jnp.asarray(a, dtype) for a in (lower, diag, upper))
    block = jax.ShapeDtypeStruct(diag.shape, dtype)
    bad = jax.ShapeDtypeStruct(diag.shape[:-3], jnp.bool_)
    inv, cp, bad = jax.ffi.ffi_call(
        f"{_PREFIX}_factor_{suffix}", (block, block, bad), vmap_method="broadcast_all"
    )(lower, diag, upper, np.asarray(offsets, np.int32))
    return SweepFactor(inv=inv, cp=cp, low=lower), bad


def chain_solve(
    factor: SweepFactor, rhs: Array, offsets: np.ndarray, transpose: bool = False
) -> Array:
    """Solve with a [chain_factor][] result.

    Args:
        factor: The [SweepFactor][] from [chain_factor][].
        rhs: Right-hand sides, shape ``(S, b, m)``.
        offsets: The ``offsets`` the factor was built with.
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
    )(inv, cp, low, rhs, np.asarray(offsets, np.int32), transpose=bool(transpose))
