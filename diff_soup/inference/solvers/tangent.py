"""Exact tiled forward-mode Jacobians.

Tangent basis directions are pushed forward in static multi-right-hand-side
tiles, each tile evaluating the primal together with its columns: one
``vmap`` over ``jax.jvp``.  Concatenating the resulting columns before
contraction preserves every cross-tile normal matrix term.

The primal is not linearized separately and replayed.  For a function that
runs an adaptive time integrator, ``jax.linearize`` cannot save the primal
trajectory; its staged push-forward integrates the primal a second time
beside the tangents, so a separately linearized primal would be paid twice.
[linearize_value][diff_soup.inference.solvers.tangent.linearize_value]
therefore materializes the Jacobian with the value, and the push-forward it
returns applies that matrix.

A tile of width ``w`` through an implicit time integrator carries ``w``
tangent columns of the state history and of every saved sample, and every
sparse solve inside the loop takes its right-hand-side-only batching case: one
factorization, ``w`` columns.  Memory therefore grows with ``w`` times the
state size, never with the square of the state.  ``tile="full"`` makes the
tile as wide as ``point``, so ``point`` should hold the parameters being
differentiated and nothing else.  Every tile evaluates the primal it
differentiates, so a narrower tile trades memory for one primal evaluation per
additional tile.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

_VALID_TILES = (8, 16, 32, 64, 128)


def normalize_tangent_tile(tile: str | int | None, width: int) -> int:
    """Resolve a public tangent-tile policy to one positive static width."""
    width = int(width)
    if width < 0:
        raise ValueError("tangent width must be nonnegative")
    if tile is None or tile == "full":
        return max(width, 1)
    try:
        resolved = int(tile)
    except (TypeError, ValueError) as exc:
        raise ValueError("tangent tile must be one of 8, 16, 32, 64, 128, or 'full'") from exc
    if resolved not in _VALID_TILES:
        raise ValueError("tangent tile must be one of 8, 16, 32, 64, 128, or 'full'")
    return min(resolved, max(width, 1))


class _MaterializedPushforward(eqx.Module):
    """The push-forward of a function whose Jacobian is already materialized."""

    jacobian: jax.Array
    out_shape: tuple = eqx.field(static=True)

    def __call__(self, tangent: Any) -> jax.Array:
        return jnp.reshape(self.jacobian @ jnp.asarray(tangent), self.out_shape)


def _tiled_value_and_jacobian(
    function: Callable[[Any], Any], point: jax.Array, tile: str | int | None
) -> tuple[jax.Array, jax.Array, tuple]:
    """Evaluate ``function`` and its Jacobian in fused primal-and-tangent tiles.

    Returns the flattened value, the ``(n_out, n_in)`` Jacobian, and the
    function's output shape.
    """
    width = int(point.shape[0])
    if width == 0:
        value = jnp.asarray(function(point))
        flat = jnp.reshape(value, (-1,))
        return flat, jnp.zeros((flat.shape[0], 0), dtype=flat.dtype), value.shape
    tile_width = normalize_tangent_tile(tile, width)
    identity = jnp.eye(width, dtype=point.dtype)
    n_tiles = (width + tile_width - 1) // tile_width
    padded_width = n_tiles * tile_width
    tiled_basis = jnp.pad(identity, ((0, padded_width - width), (0, 0))).reshape(
        n_tiles,
        tile_width,
        width,
    )

    def apply_tile(tangents):
        values, columns = jax.vmap(lambda t: jax.jvp(function, (point,), (t,)))(tangents)
        return values[0], columns

    if n_tiles == 1:
        value, columns = apply_tile(tiled_basis[0])
    else:
        values, columns = jax.lax.map(apply_tile, tiled_basis)
        value = values[0]
    value = jnp.asarray(value)
    flat = jnp.reshape(value, (-1,))
    columns = jnp.reshape(columns, (padded_width, flat.shape[0]))[:width]
    return flat, columns.T, value.shape


def linearize_value(
    function: Callable[[Any], Any], point: Any, *, tile: str | int | None = "full"
) -> tuple[jax.Array, Callable]:
    """Evaluate a vector function once and return its reusable push-forward.

    The Jacobian is materialized with the value, in tiles of ``tile`` columns;
    [jacfwd_from_pushforward][diff_soup.inference.solvers.tangent.jacfwd_from_pushforward]
    returns it without further evaluation.
    """
    point = jnp.asarray(point)
    if point.ndim != 1:
        raise ValueError(f"tiled jacfwd point must be one-dimensional; got {point.shape}")
    value, jacobian, out_shape = _tiled_value_and_jacobian(function, point, tile)
    return value, _MaterializedPushforward(jacobian, out_shape)


def deferred_value_and_grad(
    function: Callable[[Any], Any], point: Any, *, ad: str
) -> tuple[jax.Array, Callable[[], jax.Array]]:
    """Evaluate a scalar function once and defer its gradient sweep.

    With ``ad="forward"`` the gradient is computed with the value (see
    [linearize_value][diff_soup.inference.solvers.tangent.linearize_value]).
    """
    point = jnp.asarray(point)
    if point.ndim != 1:
        raise ValueError(f"gradient point must be one-dimensional; got {point.shape}")
    if ad == "reverse":
        value, pullback = jax.vjp(function, point)
        return value, lambda: pullback(jnp.ones_like(value))[0]
    if ad == "forward":
        flat, jacobian, out_shape = _tiled_value_and_jacobian(function, point, "full")
        return jnp.reshape(flat, out_shape), lambda: jacobian[0]
    raise ValueError(f"gradient differentiation mode must be 'forward' or 'reverse'; got {ad!r}.")


def jacfwd_from_pushforward(
    pushforward: Callable,
    point: Any,
    value: Any,
    *,
    tile: str | int | None = "full",
) -> jax.Array:
    """Materialize a tiled Jacobian from an existing forward linearization.

    A push-forward returned by
    [linearize_value][diff_soup.inference.solvers.tangent.linearize_value]
    already holds its Jacobian, which is returned as is; any other linear map
    is applied to the basis in tiles of ``tile`` columns.
    """
    if isinstance(pushforward, _MaterializedPushforward):
        return pushforward.jacobian
    point = jnp.asarray(point)
    value = jnp.reshape(jnp.asarray(value), (-1,))
    if point.ndim != 1:
        raise ValueError(f"tiled jacfwd point must be one-dimensional; got {point.shape}")
    width = int(point.shape[0])
    if width == 0:
        return jnp.zeros((value.shape[0], 0), dtype=value.dtype)
    tile_width = normalize_tangent_tile(tile, width)
    identity = jnp.eye(width, dtype=point.dtype)
    n_tiles = (width + tile_width - 1) // tile_width
    padded_width = n_tiles * tile_width
    tiled_basis = jnp.pad(identity, ((0, padded_width - width), (0, 0))).reshape(
        n_tiles,
        tile_width,
        width,
    )

    def apply_tile(tangents):
        return jax.vmap(pushforward)(tangents)

    directional = (
        apply_tile(tiled_basis[0])[None] if n_tiles == 1 else jax.lax.map(apply_tile, tiled_basis)
    )
    directional = jnp.reshape(directional, (padded_width, value.shape[0]))[:width]
    return directional.T


def value_and_jacfwd_tiled(
    function: Callable[[Any], Any],
    point: Any,
    *,
    tile: str | int | None = "full",
) -> tuple[jax.Array, jax.Array]:
    """Evaluate ``function`` once per tile together with its exact tiled Jacobian."""
    point = jnp.asarray(point)
    if point.ndim != 1:
        raise ValueError(f"tiled jacfwd point must be one-dimensional; got {point.shape}")
    value, jacobian, _ = _tiled_value_and_jacobian(function, point, tile)
    return value, jacobian


def value_and_jacfwd(function: Callable[[Any], Any], point: Any) -> tuple[jax.Array, jax.Array]:
    """Evaluate a vector function once together with its forward Jacobian."""
    return value_and_jacfwd_tiled(function, point, tile="full")


__all__ = [
    "deferred_value_and_grad",
    "jacfwd_from_pushforward",
    "linearize_value",
    "normalize_tangent_tile",
    "value_and_jacfwd",
    "value_and_jacfwd_tiled",
]
