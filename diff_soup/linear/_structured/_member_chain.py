"""Block-tridiagonal chains of many members at once: a sweep over the site index.

``M`` independent block chains of ``N`` sites and ``b x b`` blocks are stored
member-major, ``lower``, ``diag``, ``upper`` of shape ``(M, N, b, b)``, with
``lower[m, i]`` multiplying ``x[m, i - 1]`` and ``upper[m, i]`` multiplying
``x[m, i + 1]`` (``lower[:, 0]`` and ``upper[:, N - 1]`` are ignored). A
member shorter than ``N`` is padded with identity diagonal blocks and zero
couplings, which leaves its real sites untouched.

The sweep is block Thomas elimination, one step per site for all members at
once. It does not pivot across sites; each ``b x b`` pivot block is inverted
with partial pivoting inside the block. With ``P_i = D_i - L_i C_{i-1}`` and
``C_i = P_i^-1 U_i``,

    T = Lt Ut,   Lt = bidiag(L_i, P_i),   Ut = bidiag(I, C_i),

so ``T x = r`` is ``g_i = P_i^-1 (r_i - L_i g_{i-1})`` then ``x_i = g_i - C_i
x_{i+1}``, and ``T^T x = c`` uses the same factor: ``z_i = c_i - C_{i-1}^T
z_{i-1}`` then ``x_i = P_i^-T (z_i - L_{i+1}^T x_{i+1})``. Transposes are plain
(never conjugated), so complex chains factor and solve like real ones.

``elimination="cyclic"`` runs the same chains through block cyclic reduction
([block_tridiagonal_factor][]) as one flat ragged chain: ``ceil(log2(M N))``
levels instead of ``N`` sequential steps.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from diff_soup._errors import ConfigurationError
from diff_soup.linear._structured._block_tridiagonal import (
    block_tridiagonal_factor,
    block_tridiagonal_solve,
)
from diff_soup.linear._structured._small_lu import (
    SINGULAR_PIVOT_ROUNDOFFS,
    bmm,
    small_lu_factor,
    small_lu_solve,
)

__all__ = ["ELIMINATIONS", "SweepFactor", "chain_factor", "chain_solve"]

#: The elimination kernels a member chain can be factored with.
ELIMINATIONS = ("thomas", "cyclic", "native")


class SweepFactor(eqx.Module):
    """The Thomas factor of ``M`` chains: ``inv = P^-1``, ``cp = P^-1 U`` and ``low = L``."""

    inv: Array
    cp: Array
    low: Array


def _mul(a: Array, x: Array) -> Array:
    """``a @ x`` over the two trailing axes (a product for ``1 x 1`` blocks)."""
    return a * x if a.shape[-1] == 1 else bmm(a, x)


def _mul_t(a: Array, x: Array) -> Array:
    """``a^T @ x`` over the two trailing axes, with the plain transpose."""
    return a * x if a.shape[-1] == 1 else bmm(jnp.swapaxes(a, -1, -2), x)


def _sites_first(x: Array) -> Array:
    """Move the site axis of a member-major array ``(M, N, ...)`` to the front."""
    return jnp.moveaxis(x, 1, 0)


def _invert(pivot: Array, scale: Array) -> tuple[Array, Array]:
    """Inverse of every ``b x b`` pivot block and whether it is singular.

    ``scale`` ``(M,)`` is the magnitude of the block row before elimination,
    which a ``1 x 1`` pivot is compared with.
    """
    b = pivot.shape[-1]
    if b == 1:
        roundoff = SINGULAR_PIVOT_ROUNDOFFS * jnp.finfo(pivot.dtype).eps / 2
        p = pivot[..., 0, 0]
        bad = ~jnp.isfinite(p) | (jnp.abs(p) <= roundoff * scale)
        return 1.0 / pivot, bad
    lu, perm, bad = small_lu_factor(pivot)
    eye = jnp.broadcast_to(jnp.eye(b, dtype=pivot.dtype), pivot.shape)
    return small_lu_solve(lu, perm, eye), bad


def _sweep_factor(lower: Array, diag: Array, upper: Array) -> tuple[SweepFactor, Array]:
    """Thomas elimination over the site axis for all members at once."""
    m, _, b, _ = diag.shape
    scale = jnp.max(jnp.abs(lower) + jnp.abs(diag) + jnp.abs(upper), axis=(-1, -2))

    def step(carry, site):
        cp_prev, bad = carry
        low, d, up, s = site
        inv, singular = _invert(d - _mul(low, cp_prev), s)
        cp = _mul(inv, up)
        return (cp, bad | singular), (inv, cp)

    first = jnp.zeros((m, b, b), diag.dtype)
    # The first site has no predecessor: its lower block is ignored.
    low_used = lower.at[:, 0].set(0)
    (_, bad), (inv, cp) = jax.lax.scan(
        step,
        (first, jnp.zeros((m,), bool)),
        (_sites_first(low_used), _sites_first(diag), _sites_first(upper), _sites_first(scale)),
    )
    factor = SweepFactor(inv=jnp.moveaxis(inv, 0, 1), cp=jnp.moveaxis(cp, 0, 1), low=low_used)
    return factor, bad


def _sweep_solve(factor: SweepFactor, rhs: Array, transpose: bool) -> Array:
    """``T^-1 rhs`` (or ``T^-T rhs``) for ``rhs`` ``(M, N, b, m)`` from a [SweepFactor][]."""
    inv, cp, low = (_sites_first(x) for x in (factor.inv, factor.cp, factor.low))
    r = _sites_first(rhs)
    zero = jnp.zeros(r.shape[1:], r.dtype)
    if not transpose:

        def down(g_prev, site):
            inv_i, low_i, r_i = site
            g = _mul(inv_i, r_i - _mul(low_i, g_prev))
            return g, g

        _, g = jax.lax.scan(down, zero, (inv, low, r))

        def up(x_next, site):
            cp_i, g_i = site
            x = g_i - _mul(cp_i, x_next)
            return x, x

        _, x = jax.lax.scan(up, zero, (cp, g), reverse=True)
    else:
        # C_{i-1} and L_{i+1}: shift the stored blocks by one site.
        cp_prev = jnp.concatenate([jnp.zeros_like(cp[:1]), cp[:-1]], axis=0)
        low_next = jnp.concatenate([low[1:], jnp.zeros_like(low[:1])], axis=0)

        def down(z_prev, site):
            cp_i, c_i = site
            z = c_i - _mul_t(cp_i, z_prev)
            return z, z

        _, z = jax.lax.scan(down, zero, (cp_prev, r))

        def up(x_next, site):
            inv_i, low_i, z_i = site
            x = _mul_t(inv_i, z_i - _mul_t(low_i, x_next))
            return x, x

        _, x = jax.lax.scan(up, zero, (inv, low_next, z), reverse=True)
    return jnp.moveaxis(x, 0, 1)


def _flat(x: Array) -> Array:
    """Members stored back to back: ``(M, N, ...)`` as one chain ``(M N, ...)``."""
    return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def chain_factor(
    lower: Array, diag: Array, upper: Array, elimination: str = "thomas"
) -> tuple[Any, Array]:
    """Factor ``M`` block chains.

    Args:
        lower: ``(M, N, b, b)`` blocks multiplying the previous site.
        diag: ``(M, N, b, b)`` diagonal blocks.
        upper: ``(M, N, b, b)`` blocks multiplying the next site.
        elimination: One of [ELIMINATIONS][].

    Returns:
        ``(factor, bad)``: the factor (a pytree of arrays that depends on
        ``elimination``) and a boolean ``(M,)`` that holds where a member has
        a negligible or non-finite pivot.

    Raises:
        ConfigurationError: For an unknown ``elimination``.
    """
    if elimination == "thomas":
        return _sweep_factor(lower, diag, upper)
    if elimination == "cyclic":
        m, n = diag.shape[:2]
        low = lower.at[:, 0].set(0)
        up = upper.at[:, n - 1].set(0)
        factor, bad = block_tridiagonal_factor(_flat(low), _flat(diag), _flat(up))
        return factor, jnp.broadcast_to(bad, (m,))
    if elimination == "native":
        from diff_soup.linear._structured import _member_chain_native as native

        return native.chain_factor(lower, diag, upper)
    raise ConfigurationError(f"elimination must be one of {ELIMINATIONS}; got {elimination!r}.")


def chain_solve(
    factor: Any, rhs: Array, transpose: bool = False, elimination: str = "thomas"
) -> Array:
    """Solve ``T x = rhs`` (or ``T^T x = rhs``) for ``M`` chains from a [chain_factor][] result.

    Args:
        factor: The factor [chain_factor][] returned for the same ``elimination``.
        rhs: ``(M, N, b, m)`` right-hand sides.
        transpose: Solve with the plain transpose from the same factor.
        elimination: The kernel the factor was built with.

    Returns:
        ``x`` with the shape of ``rhs``.

    Raises:
        ConfigurationError: For an unknown ``elimination``.
    """
    if elimination == "thomas":
        return _sweep_solve(factor, rhs, transpose)
    if elimination == "cyclic":
        return block_tridiagonal_solve(factor, _flat(rhs), transpose=transpose).reshape(rhs.shape)
    if elimination == "native":
        from diff_soup.linear._structured import _member_chain_native as native

        return native.chain_solve(factor, rhs, transpose)
    raise ConfigurationError(f"elimination must be one of {ELIMINATIONS}; got {elimination!r}.")
