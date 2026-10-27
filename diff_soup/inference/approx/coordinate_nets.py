"""Fourier-feature and sinusoidal networks for physical coordinates."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def _activation(name):
    """Resolve supported activation names or a supplied callable."""
    functions = {
        "tanh": jnp.tanh,
        "softplus": jax.nn.softplus,
        "relu": jax.nn.relu,
        "sigmoid": jax.nn.sigmoid,
    }
    if callable(name):
        return name
    if name not in functions:
        raise ValueError(f"activation must be callable or one of {tuple(functions)}; got {name!r}")
    return functions[name]


def _array_paths(module):
    """List nested fitted array paths in the public dotted-path grammar."""
    paths = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(module):
        if eqx.is_inexact_array(leaf):
            paths.append(
                ".".join(
                    str(
                        key.name
                        if hasattr(key, "name")
                        else key.idx
                        if hasattr(key, "idx")
                        else key.key
                    )
                    for key in path
                )
            )
    return tuple(paths)


def _bounds(bounds, in_size):
    """Validate fixed affine-normalization bounds."""
    if bounds is None:
        return None
    bounds = tuple(tuple(map(float, interval)) for interval in bounds)
    if len(bounds) != in_size or any(
        len(b) != 2 or not np.isfinite(b).all() or b[0] >= b[1] for b in bounds
    ):
        raise ValueError(f"bounds must contain {in_size} finite (lo, hi) pairs with lo < hi")
    return bounds


def _coordinates(x, in_size, bounds):
    """Normalize coordinates, accepting a one-dimensional grid vector too."""
    x = jnp.asarray(x)
    if in_size == 1 and (x.ndim == 0 or x.shape[-1] != 1):
        x = x[..., None]
    if x.ndim == 0 or x.shape[-1] != in_size:
        raise ValueError(f"coordinates must have trailing dimension {in_size}; got shape {x.shape}")
    if bounds is None:
        return x
    limits = jnp.asarray(bounds)
    return 2 * (x - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1


def _batched_mlp(mlp, x, out_size):
    """Apply a sample-wise MLP across every leading coordinate axis."""
    value = jax.vmap(mlp)(x.reshape((-1, x.shape[-1])))
    value = value.reshape(x.shape[:-1] + (out_size,))
    return value[..., 0] if out_size == 1 else value


class FourierFeatureNet(eqx.Module):
    """A coordinate network with fixed sine and cosine input features.

    Every coordinate contributes sin(2**k pi x) and cos(2**k pi x) for
    k=0,...,n_frequencies-1. Bounds normalize physical coordinates to [-1, 1].
    Only the MLP weights and biases are fitted. ``transform(x, raw)`` may
    impose output structure, for example ``lambda x, raw: exp(raw)``.
    """

    mlp: eqx.nn.MLP
    frequencies: tuple = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    bounds: Any = eqx.field(static=True)
    transform: Any = eqx.field(static=True)

    def __init__(
        self,
        in_size,
        *,
        n_frequencies=8,
        width=32,
        depth=2,
        key,
        out_size=1,
        activation="tanh",
        transform=None,
        bounds=None,
    ):
        """Initialize fixed frequencies and a fitted MLP on their features."""
        if min(in_size, out_size, width, n_frequencies) < 1 or depth < 0:
            raise ValueError(
                "network sizes and n_frequencies must be positive, and depth nonnegative"
            )
        self.in_size, self.out_size = in_size, out_size
        self.bounds, self.transform = _bounds(bounds, in_size), transform
        self.frequencies = tuple(float(2**k * np.pi) for k in range(n_frequencies))
        self.mlp = eqx.nn.MLP(
            2 * in_size * n_frequencies,
            out_size,
            width,
            depth,
            activation=_activation(activation),
            key=key,
        )

    @classmethod
    def from_domain(cls, domain, **kwargs):
        """Take input dimension and normalization bounds from a domain."""
        return cls(domain.dim, bounds=domain.bounds, **kwargs)

    def __call__(self, x):
        """Evaluate one coordinate or a batch, squeezing a scalar output channel."""
        normalized = _coordinates(x, self.in_size, self.bounds)
        phase = normalized[..., :, None] * jnp.asarray(self.frequencies)
        features = jnp.concatenate((jnp.sin(phase), jnp.cos(phase)), axis=-1)
        features = features.reshape(normalized.shape[:-1] + (-1,))
        raw = _batched_mlp(self.mlp, features, self.out_size)
        return raw if self.transform is None else self.transform(x, raw)

    def learnable_paths(self) -> tuple[str, ...]:
        """Return fitted MLP leaves; fixed frequencies are not learnable."""
        return tuple("mlp." + path for path in _array_paths(self.mlp))


class SIREN(eqx.Module):
    """Sinusoidal coordinate network with variance-preserving initialization.

    Hidden layers apply sin(omega_0 * z); the output layer is linear.
    The first weight matrix is uniform on [-1/in_size, 1/in_size], and
    later matrices on +/-sqrt(6/fan_in)/omega_0. Bounds and the optional
    ``transform(x, raw)`` have the same meaning as in FourierFeatureNet.
    """

    mlp: eqx.nn.MLP
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    omega_0: float = eqx.field(static=True)
    bounds: Any = eqx.field(static=True)
    transform: Any = eqx.field(static=True)

    def __init__(
        self,
        in_size,
        *,
        width=32,
        depth=3,
        key,
        out_size=1,
        omega_0=30.0,
        transform=None,
        bounds=None,
    ):
        """Initialize sine layers and the final linear readout."""
        if min(in_size, out_size, width, depth) < 1 or omega_0 <= 0:
            raise ValueError("SIREN sizes, depth, and omega_0 must be positive")
        self.in_size, self.out_size, self.omega_0 = in_size, out_size, float(omega_0)
        self.bounds, self.transform = _bounds(bounds, in_size), transform
        keys = jax.random.split(key, depth + 2)
        mlp = eqx.nn.MLP(
            in_size, out_size, width, depth, key=keys[0], activation=lambda z: jnp.sin(omega_0 * z)
        )
        for index, layer in enumerate(mlp.layers):
            limit = 1 / in_size if index == 0 else np.sqrt(6 / layer.weight.shape[1]) / omega_0
            weight = jax.random.uniform(
                keys[index + 1], layer.weight.shape, minval=-limit, maxval=limit
            )
            mlp = eqx.tree_at(lambda m: m.layers[index].weight, mlp, weight)
        self.mlp = mlp

    @classmethod
    def from_domain(cls, domain, **kwargs):
        """Take input dimension and normalization bounds from a domain."""
        return cls(domain.dim, bounds=domain.bounds, **kwargs)

    def __call__(self, x):
        """Evaluate one coordinate or a batch with scalar-channel squeezing."""
        normalized = _coordinates(x, self.in_size, self.bounds)
        raw = _batched_mlp(self.mlp, normalized, self.out_size)
        return raw if self.transform is None else self.transform(x, raw)

    def learnable_paths(self) -> tuple[str, ...]:
        """Return the fitted weight and bias paths in every layer."""
        return tuple("mlp." + path for path in _array_paths(self.mlp))
