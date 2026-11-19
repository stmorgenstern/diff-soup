"""Block-tridiagonal chains of many members, stored back to back with no padding.

``M`` independent block chains (members) of ``n_m`` sites and ``b x b`` blocks
are one flat chain of ``S = sum(n_m)`` sites: ``lower``, ``diag``, ``upper``
of shape ``(S, b, b)``, with ``lower[p]`` multiplying the site before ``p``
and ``upper[p]`` the site after it. ``lower`` at the first site of a member
and ``upper`` at its last site must be zero, so members never exchange
information. A [ChainLayout][] holds the members' lengths and the index maps
that depend on them.

Three kernels factor and solve the same chains:

- ``"cyclic"``: block cyclic reduction of the flat chain
  ([block_tridiagonal_factor][]), ``ceil(log2(S))`` levels of work that halves
  each level. Nothing is padded.
- ``"thomas"``: block Thomas elimination, one step per site index for all
  members at once. With ``P_i = D_i - L_i C_{i-1}`` and ``C_i = P_i^-1 U_i``,
  ``T x = r`` is ``g_i = P_i^-1 (r_i - L_i g_{i-1})`` then ``x_i = g_i - C_i
  x_{i+1}``, and ``T^T x = c`` uses the same factor. Members of different
  lengths are gathered to the longest one for the sweep and gathered back;
  the factor is stored flat.
- ``"native"``: the same Thomas elimination in the threaded processor kernel,
  each member at its own length.

No kernel pivots across sites; each ``b x b`` pivot block is inverted with
partial pivoting inside the block. Transposes are plain (never conjugated),
so complex chains factor and solve like real ones.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from diff_soup._errors import ConfigurationError
from diff_soup.linear._structured._block_tridiagonal import (
    block_tridiagonal_factor,
    block_tridiagonal_solve,
)
from diff_soup.linear._structured._common import index_constant
from diff_soup.linear._structured._small_lu import (
    SINGULAR_PIVOT_ROUNDOFFS,
    bmm,
    small_lu_factor,
    small_lu_solve,
)

__all__ = ["ELIMINATIONS", "ChainLayout", "SweepFactor", "chain_factor", "chain_solve"]

#: The elimination kernels a member chain can be factored with.
ELIMINATIONS = ("thomas", "cyclic", "native")


class ChainLayout:
    """The lengths of the members of a flat chain and the host index maps they fix."""

    def __init__(self, lengths: Any):
        """Record the lengths; the maps are built when a kernel first asks for them."""
        lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
        self.lengths = lengths
        self.n_members = int(lengths.size)
        self.n_sites = int(lengths.sum())
        self.n_max = int(lengths.max())
        self.uniform = bool(np.all(lengths == self.n_max))
        self._maps: dict = {}

    @property
    def offsets(self) -> np.ndarray:
        """``(M + 1,)`` first site of every member, and the site count (host int32)."""
        if "offsets" not in self._maps:
            self._maps["offsets"] = index_constant(np.concatenate([[0], np.cumsum(self.lengths)]))
        return self._maps["offsets"]

    @property
    def member_of_site(self) -> np.ndarray:
        """``(S,)`` the member of every site (host int32)."""
        if "member" not in self._maps:
            self._maps["member"] = index_constant(np.repeat(np.arange(self.n_members), self.lengths))
        return self._maps["member"]

    @property
    def pad(self) -> np.ndarray:
        """``(M, n_max)`` the flat site of every padded place, ``S`` where a member has none."""
        if "pad" not in self._maps:
            start = np.asarray(self.offsets[:-1], dtype=np.int64)
            place = start[:, None] + np.arange(self.n_max)[None, :]
            real = np.arange(self.n_max)[None, :] < self.lengths[:, None]
            self._maps["pad"] = index_constant(np.where(real, place, self.n_sites))
        return self._maps["pad"]

    @property
    def flat(self) -> np.ndarray:
        """``(S,)`` the padded place ``m * n_max + i`` of every flat site."""
        if "flat" not in self._maps:
            start = np.asarray(self.offsets[:-1], dtype=np.int64)
            within = np.arange(self.n_sites) - np.repeat(start, self.lengths)
            self._maps["flat"] = index_constant(
                np.repeat(np.arange(self.n_members), self.lengths) * self.n_max + within
            )
        return self._maps["flat"]

    def padded(self, x: Array, unit: bool = False) -> Array:
        """``x`` ``(S, ...)`` member-major ``(M, n_max, ...)``; padded places are zero, or identity blocks."""
        shape = (self.n_members, self.n_max) + x.shape[1:]
        if self.uniform:
            return x.reshape(shape)
        fill = jnp.zeros((1,) + x.shape[1:], x.dtype)
        if unit:
            fill = fill + jnp.eye(x.shape[-1], dtype=x.dtype)
        return jnp.concatenate([x, fill], axis=0)[self.pad]

    def flattened(self, x: Array) -> Array:
        """``x`` ``(M, n_max, ...)`` as the flat ``(S, ...)`` of its real sites."""
        x = x.reshape((self.n_members * self.n_max,) + x.shape[2:])
        return x if self.uniform else x[self.flat]


class SweepFactor(eqx.Module):
    """The Thomas factor of a flat chain: ``inv = P^-1``, ``cp = P^-1 U`` and ``low = L``, each ``(S, b, b)``."""

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


def _sweep_factor(lower: Array, diag: Array, upper: Array) -> tuple[Array, Array, Array]:
    """Thomas elimination over the site axis of member-major ``(M, N, b, b)`` blocks.

    Returns:
        ``(inv, cp, bad)``, the first two member-major and ``bad`` one boolean.
    """
    m, _, b, _ = diag.shape
    scale = jnp.max(jnp.abs(lower) + jnp.abs(diag) + jnp.abs(upper), axis=(-1, -2))

    def step(carry, site):
        cp_prev, bad = carry
        low, d, up, s = site
        inv, singular = _invert(d - _mul(low, cp_prev), s)
        cp = _mul(inv, up)
        return (cp, bad | singular), (inv, cp)

    first = jnp.zeros((m, b, b), diag.dtype)
    (_, bad), (inv, cp) = jax.lax.scan(
        step,
        (first, jnp.zeros((m,), bool)),
        (_sites_first(lower), _sites_first(diag), _sites_first(upper), _sites_first(scale)),
    )
    return jnp.moveaxis(inv, 0, 1), jnp.moveaxis(cp, 0, 1), jnp.any(bad)


def _sweep_solve(inv: Array, cp: Array, low: Array, rhs: Array, transpose: bool) -> Array:
    """``T^-1 rhs`` (or ``T^-T rhs``) for member-major ``rhs`` ``(M, N, b, m)``."""
    inv, cp, low = (_sites_first(x) for x in (inv, cp, low))
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


def chain_factor(
    lower: Array, diag: Array, upper: Array, layout: ChainLayout, elimination: str = "thomas"
) -> tuple[Any, Array]:
    """Factor the members of a flat chain.

    Args:
        lower: ``(S, b, b)`` blocks multiplying the previous site, zero at a member's first site.
        diag: ``(S, b, b)`` diagonal blocks.
        upper: ``(S, b, b)`` blocks multiplying the next site, zero at a member's last site.
        layout: The members' lengths.
        elimination: One of [ELIMINATIONS][].

    Returns:
        ``(factor, bad)``: the factor (a pytree of arrays that depends on
        ``elimination``) and one boolean that holds where some member has a
        negligible or non-finite pivot.

    Raises:
        ConfigurationError: For an unknown ``elimination``.
    """
    if elimination == "thomas":
        inv, cp, bad = _sweep_factor(
            layout.padded(lower), layout.padded(diag, unit=True), layout.padded(upper)
        )
        return SweepFactor(inv=layout.flattened(inv), cp=layout.flattened(cp), low=lower), bad
    if elimination == "cyclic":
        factor, bad = block_tridiagonal_factor(lower, diag, upper)
        return factor, bad
    if elimination == "native":
        from diff_soup.linear._structured import _member_chain_native as native

        return native.chain_factor(lower, diag, upper, layout.offsets)
    raise ConfigurationError(f"elimination must be one of {ELIMINATIONS}; got {elimination!r}.")


def chain_solve(
    factor: Any,
    rhs: Array,
    layout: ChainLayout,
    transpose: bool = False,
    elimination: str = "thomas",
) -> Array:
    """Solve ``T x = rhs`` (or ``T^T x = rhs``) for a flat chain from a [chain_factor][] result.

    Args:
        factor: The factor [chain_factor][] returned for the same ``elimination``.
        rhs: ``(S, b, m)`` right-hand sides.
        layout: The members' lengths.
        transpose: Solve with the plain transpose from the same factor.
        elimination: The kernel the factor was built with.

    Returns:
        ``x`` with the shape of ``rhs``.

    Raises:
        ConfigurationError: For an unknown ``elimination``.
    """
    if elimination == "thomas":
        x = _sweep_solve(
            layout.padded(factor.inv),
            layout.padded(factor.cp),
            layout.padded(factor.low),
            layout.padded(rhs),
            transpose,
        )
        return layout.flattened(x)
    if elimination == "cyclic":
        return block_tridiagonal_solve(factor, rhs, transpose=transpose)
    if elimination == "native":
        from diff_soup.linear._structured import _member_chain_native as native

        return native.chain_solve(factor, rhs, layout.offsets, transpose)
    raise ConfigurationError(f"elimination must be one of {ELIMINATIONS}; got {elimination!r}.")
