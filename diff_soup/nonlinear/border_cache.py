"""Generation-keyed cache of bordered-system Schur artifacts.

A corrector for a system with bordered-global rows eliminates the state
block each Newton sweep: with state factor ``A`` and coupling
blocks ``B`` / ``C`` / ``D`` it forms the border solve ``du_b = A^{-1} B``,
the Schur complement ``S = D - C @ du_b``, and then solves the small dense
system ``S dlam = rhs`` once per Newton iteration. All three artifacts are
pure functions of the carried linearization, so between rebuild events they
can be reused instead of recomputed.

[BorderSchurCache][] holds the three artifacts with the generation keys
they were built from (see [diff_soup.nonlinear.generations][]):

* ``du_b`` is keyed by the state-factor generation, the ``B``-block
  generation, and the active linear-precision rung;
* ``S`` is keyed by the cache's own ``du_b`` build counter and the ``C`` /
  ``D`` generations;
* the LU factor of the (regularized) ``S`` is keyed by the cache's own ``S``
  build counter, so a Newton iteration performs only the triangular solves
  of [border_cache_solve][] — never a refactorization.

[refresh_border_cache][] revalidates every tier against the caller's
current generation counters and rebuilds exactly the stale tiers, reporting
what it rebuilt so the caller can account the work. A fresh record from
[initial_border_cache][] carries sentinel keys that match no counter, so
a rebuilt carry (a new solve, or an event restart that reconstructs the
engine state) always starts cold. Every field is a plain array leaf: under
``vmap`` each batch lane keys and invalidates its own cache independently,
and the record composes with ``jit``, ``lax.scan``, and checkpoint replay.

The cache never participates in differentiation: correctors consult it only
inside primal solves that sit behind a root derivative boundary, and the
sensitivity rules rebuild exact values at the accepted root.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from diff_soup.utils.lanes import lane_cond

__all__ = [
    "BorderSchurCache",
    "border_cache_solve",
    "initial_border_cache",
    "refresh_border_cache",
]

_COLD_KEY = -1
"""Sentinel key on a fresh record; generation counters start at zero, so a
cold tier can never validate against a live carry."""


def _key(value) -> jax.Array:
    """Coerce a cache key to an int32 scalar."""
    return jnp.asarray(value, dtype=jnp.int32)


class BorderSchurCache(eqx.Module):
    """The cached border solve, Schur complement, and Schur LU factor.

    Each tier stores the artifact next to the generation keys it was built
    from; a tier is valid exactly when its stored keys equal the caller's
    current counters. The record's own build counters (``du_b_generation``,
    ``schur_generation``) name the cached values so downstream tiers can key
    on them.

    Attributes:
        du_b: The border solve ``A^{-1} B``, shape ``(n_state, n_border)``.
        schur: The Schur complement ``D - C @ du_b``, shape
            ``(n_border, n_border)``.
        schur_lu: The packed LU factor of the regularized Schur complement,
            shape ``(n_border, n_border)``.
        schur_piv: The LU pivot indices, shape ``(n_border,)`` (``int32``).
        du_b_factor_key: State-factor generation ``du_b`` was built against.
        du_b_b_key: ``B``-block generation ``du_b`` was built against.
        du_b_rung_key: Linear-precision rung ``du_b`` was built at (``0``
            for solve modes without a rung ladder).
        du_b_generation: Build counter of the cached ``du_b``.
        schur_du_b_key: ``du_b`` build counter ``schur`` was built against.
        schur_c_key: ``C``-block generation ``schur`` was built against.
        schur_d_key: ``D``-block generation ``schur`` was built against.
        schur_generation: Build counter of the cached ``schur``.
        factor_schur_key: ``schur`` build counter the LU factor was built
            against (the factor tier's validity key).
    """

    du_b: jax.Array
    schur: jax.Array
    schur_lu: jax.Array
    schur_piv: jax.Array
    du_b_factor_key: jax.Array
    du_b_b_key: jax.Array
    du_b_rung_key: jax.Array
    du_b_generation: jax.Array
    schur_du_b_key: jax.Array
    schur_c_key: jax.Array
    schur_d_key: jax.Array
    schur_generation: jax.Array
    factor_schur_key: jax.Array


def initial_border_cache(n_state: int, n_border: int, dtype) -> BorderSchurCache:
    """A cold cache record for a bordered system.

    Every key is the sentinel ``-1`` and the build counters are zero, so no
    tier validates until [refresh_border_cache][] builds it. Build one
    per solver carry; a carry rebuilt from scratch (a fresh solve, or an
    event restart) must take a fresh record so nothing survives the restart.

    Args:
        n_state: State-block dimension (rows of ``B``).
        n_border: Bordered-global dimension.
        dtype: Floating dtype of the cached arrays.

    Returns:
        The cold [BorderSchurCache][].
    """
    cold = _key(_COLD_KEY)
    zero = _key(0)
    return BorderSchurCache(
        du_b=jnp.zeros((n_state, n_border), dtype=dtype),
        schur=jnp.zeros((n_border, n_border), dtype=dtype),
        schur_lu=jnp.zeros((n_border, n_border), dtype=dtype),
        schur_piv=jnp.zeros((n_border,), dtype=jnp.int32),
        du_b_factor_key=cold,
        du_b_b_key=cold,
        du_b_rung_key=cold,
        du_b_generation=zero,
        schur_du_b_key=cold,
        schur_c_key=cold,
        schur_d_key=cold,
        schur_generation=zero,
        factor_schur_key=cold,
    )


def refresh_border_cache(
    cache: BorderSchurCache,
    *,
    factor_generation,
    b_generation,
    c_generation,
    d_generation,
    rung=None,
    solve_many,
    b_block: jax.Array,
    c_block: jax.Array,
    d_block: jax.Array,
    regularization: float,
):
    """Revalidate every cache tier and rebuild exactly the stale ones.

    Validity is generation-keyed: ``du_b`` is stale when the state-factor
    generation, the ``B`` generation, or the precision rung moved since it
    was built; ``schur`` (and with it the LU factor) is stale when the
    cached ``du_b`` was rebuilt or the ``C`` / ``D`` generations moved. A
    valid tier is returned untouched — no border multi-RHS solve, Schur
    assembly, or factorization runs for it. The rebuild of a stale tier uses
    the supplied blocks and solver, so the caller must pass the blocks the
    current generation counters describe.

    The LU factor is taken of ``schur + regularization * I`` — the same
    regularized matrix the correctors' small-system solve uses — so
    [border_cache_solve][] reproduces that solve exactly.

    Args:
        cache: The carried [BorderSchurCache][].
        factor_generation: The state-factor generation counter naming the
            factor behind ``solve_many``.
        b_generation: The ``B``-block generation counter.
        c_generation: The ``C``-block generation counter.
        d_generation: The ``D``-block generation counter.
        rung: The active linear-precision rung (traced ``int32``), or
            ``None`` for solve modes without a rung ladder.
        solve_many: Multi-right-hand-side solve against the current state
            factor, mapping ``(nrhs, n_state)`` rows to solutions.
        b_block: The current ``B`` block, shape ``(n_state, n_border)``.
        c_block: The current ``C`` block, shape ``(n_border, n_state)``.
        d_block: The current ``D`` block, shape ``(n_border, n_border)``.
        regularization: Tikhonov magnitude added to ``schur`` before
            factoring.

    Returns:
        ``(cache, du_b_built, schur_built)`` — the fully valid record and
        two ``int32`` 0/1 flags reporting whether the border solve and the
        Schur complement (with its factor) were rebuilt.
    """
    factor_generation = _key(factor_generation)
    b_generation = _key(b_generation)
    c_generation = _key(c_generation)
    d_generation = _key(d_generation)
    rung_key = _key(0) if rung is None else _key(rung)

    du_b_valid = (
        (cache.du_b_factor_key == factor_generation)
        & (cache.du_b_b_key == b_generation)
        & (cache.du_b_rung_key == rung_key)
    )

    def _rebuild_du_b(c: BorderSchurCache) -> BorderSchurCache:
        """Solve the border columns against the current factor and re-key."""
        du_b = solve_many(b_block.T).T
        return eqx.tree_at(
            lambda r: (r.du_b, r.du_b_factor_key, r.du_b_b_key, r.du_b_rung_key, r.du_b_generation),
            c,
            (du_b, factor_generation, b_generation, rung_key, c.du_b_generation + 1),
        )

    cache = lane_cond(
        jnp.logical_not(du_b_valid), _rebuild_du_b, lambda c: c, cache, false_is_free=True
    )

    schur_valid = (
        (cache.schur_du_b_key == cache.du_b_generation)
        & (cache.schur_c_key == c_generation)
        & (cache.schur_d_key == d_generation)
    )

    def _rebuild_schur(c: BorderSchurCache) -> BorderSchurCache:
        """Assemble the Schur complement, factor it, and re-key both tiers."""
        schur = d_block - c_block @ c.du_b
        n_border = schur.shape[0]
        eye = jnp.eye(n_border, dtype=schur.dtype)
        reg = jnp.asarray(regularization, dtype=schur.dtype)
        lu, piv = jax.scipy.linalg.lu_factor(schur + reg * eye)
        schur_generation = c.schur_generation + 1
        return eqx.tree_at(
            lambda r: (
                r.schur,
                r.schur_lu,
                r.schur_piv,
                r.schur_du_b_key,
                r.schur_c_key,
                r.schur_d_key,
                r.schur_generation,
                r.factor_schur_key,
            ),
            c,
            (
                schur,
                lu,
                piv.astype(jnp.int32),
                c.du_b_generation,
                c_generation,
                d_generation,
                schur_generation,
                schur_generation,
            ),
        )

    cache = lane_cond(
        jnp.logical_not(schur_valid), _rebuild_schur, lambda c: c, cache, false_is_free=True
    )
    du_b_built = jnp.logical_not(du_b_valid).astype(jnp.int32)
    schur_built = jnp.logical_not(schur_valid).astype(jnp.int32)
    return cache, du_b_built, schur_built


def border_cache_solve(cache: BorderSchurCache, rhs: jax.Array) -> jax.Array:
    """Solve the cached (regularized) Schur system by triangular solves only.

    Uses the held LU factor — no factorization runs here, so a Newton
    iteration against a valid cache costs exactly one pair of triangular
    solves. Numerically identical to an LU solve of the same regularized
    Schur complement assembled fresh.

    Args:
        cache: A record whose Schur tiers were made valid by
            [refresh_border_cache][].
        rhs: Right-hand side, shape ``(n_border,)``.

    Returns:
        The solution, shape ``(n_border,)`` (zeros when the border is empty).
    """
    if cache.schur.shape[0] == 0:
        return jnp.zeros_like(rhs)
    return jax.scipy.linalg.lu_solve((cache.schur_lu, cache.schur_piv), rhs)
