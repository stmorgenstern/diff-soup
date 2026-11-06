"""Branch/trunk and Fourier neural operators over sampled fields."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .coordinate_nets import _array_paths, _batched_mlp, _coordinates


class DeepONet(eqx.Module):
    """Branch/trunk operator mapping sensor values and coordinates to a field.

    The branch encodes ``u`` at branch_size fixed sensors. The trunk encodes
    each output coordinate of trailing size trunk_size. Their n_basis features
    are paired by an inner product plus a fitted scalar bias. Sensor batches
    may share one coordinate array or supply matching coordinate batches.
    """

    branch: eqx.nn.MLP
    trunk: eqx.nn.MLP
    bias: jax.Array
    branch_size: int = eqx.field(static=True)
    trunk_size: int = eqx.field(static=True)
    n_basis: int = eqx.field(static=True)

    def __init__(self, branch_size, trunk_size, *, width=64, depth=3, key, n_basis=None):
        """Initialize branch and trunk MLPs with a common feature count."""
        n_basis = width if n_basis is None else n_basis
        if min(branch_size, trunk_size, width, n_basis) < 1 or depth < 0:
            raise ValueError("DeepONet sizes must be positive and depth nonnegative")
        branch_key, trunk_key = jax.random.split(key)
        self.branch = eqx.nn.MLP(
            branch_size, n_basis, width, depth, key=branch_key, activation=jnp.tanh
        )
        self.trunk = eqx.nn.MLP(
            trunk_size, n_basis, width, depth, key=trunk_key, activation=jnp.tanh
        )
        self.bias = jnp.asarray(0.0)
        self.branch_size, self.trunk_size, self.n_basis = branch_size, trunk_size, n_basis

    def __call__(self, u, x):
        """Evaluate the output field at x for one or more sensor vectors u."""
        u = jnp.asarray(u)
        if u.ndim < 1 or u.shape[-1] != self.branch_size:
            raise ValueError(
                f"DeepONet sensor values need trailing size {self.branch_size}; got {u.shape}"
            )
        x = _coordinates(x, self.trunk_size, None)
        branch = _batched_mlp(self.branch, u, self.n_basis)
        trunk = _batched_mlp(self.trunk, x, self.n_basis)
        # Keep the feature axis even when the basis has only one function.
        if self.n_basis == 1:
            branch, trunk = branch[..., None], trunk[..., None]
        return jnp.sum(branch[..., None, :] * trunk, axis=-1) + self.bias

    def learnable_paths(self) -> tuple[str, ...]:
        """Return branch, trunk, and scalar-bias fitted paths."""
        return _array_paths(self)


class _SpectralLayer(eqx.Module):
    """Truncated Fourier convolution plus a pointwise channel map."""

    weights: jax.Array
    negative_weights: jax.Array | None
    pointwise: eqx.nn.Linear
    modes: tuple = eqx.field(static=True)

    def __init__(self, width, modes, key):
        first, second, local = jax.random.split(key, 3)
        shape = (width, width, *modes, 2)
        self.weights = jax.random.normal(first, shape) / width
        self.negative_weights = (
            None if len(modes) == 1 else jax.random.normal(second, shape) / width
        )
        self.pointwise = eqx.nn.Linear(width, width, key=local)
        self.modes = modes

    def __call__(self, u):
        """Preserve the spatial shape while mixing retained Fourier modes."""
        weights = self.weights[..., 0] + 1j * self.weights[..., 1]
        if len(self.modes) == 1:
            transformed = jnp.fft.rfft(u, axis=-2)
            count = min(self.modes[0], transformed.shape[-2])
            kept = jnp.einsum("...ki,iok->...ko", transformed[..., :count, :], weights[..., :count])
            result = jnp.zeros_like(transformed).at[..., :count, :].set(kept)
            spectral = jnp.fft.irfft(result, n=u.shape[-2], axis=-2)
        else:
            transformed = jnp.fft.rfft2(u, axes=(-3, -2))
            first = min(self.modes[0], (u.shape[-3] + 1) // 2)
            second = min(self.modes[1], transformed.shape[-2])
            kept = jnp.einsum(
                "...kli,iokl->...klo",
                transformed[..., :first, :second, :],
                weights[..., :first, :second],
            )
            result = jnp.zeros_like(transformed).at[..., :first, :second, :].set(kept)
            negative = min(self.modes[0], u.shape[-3] // 2)
            if negative:
                weights_negative = (
                    self.negative_weights[..., 0] + 1j * self.negative_weights[..., 1]
                )
                kept = jnp.einsum(
                    "...kli,iokl->...klo",
                    transformed[..., -negative:, :second, :],
                    weights_negative[..., :negative, :second],
                )
                result = result.at[..., -negative:, :second, :].set(kept)
            spectral = jnp.fft.irfft2(result, s=u.shape[-3:-1], axes=(-3, -2))
        return jax.nn.gelu(spectral + u @ self.pointwise.weight.T + self.pointwise.bias)


class _FNO(eqx.Module):
    """Shared channel-last Fourier-operator implementation."""

    lifting: eqx.nn.Linear
    layers: tuple
    projection: eqx.nn.Linear
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    spatial_dim: ClassVar[int]

    def __init__(self, *, modes, width, depth, key, in_channels=1, out_channels=1):
        """Build lifting, spectral layers, and projection for a uniform grid."""
        modes = (modes,) * self.spatial_dim if isinstance(modes, int) else tuple(modes)
        if len(modes) != self.spatial_dim or any(not isinstance(m, int) or m < 1 for m in modes):
            raise ValueError(
                f"FNO modes must specify {self.spatial_dim} positive integer frequency counts"
            )
        if min(width, depth, in_channels, out_channels) < 1:
            raise ValueError("FNO width, depth, and channel counts must be positive")
        keys = jax.random.split(key, depth + 2)
        self.lifting = eqx.nn.Linear(in_channels, width, key=keys[0])
        self.layers = tuple(_SpectralLayer(width, modes, key) for key in keys[1:-1])
        self.projection = eqx.nn.Linear(width, out_channels, key=keys[-1])
        self.in_channels, self.out_channels = in_channels, out_channels

    def _validate_domain(self, domain, *, name):
        """Reject geometry that is not a uniform grid of the operator's dimension."""
        from diff_soup.domains.grid.core import Grid1D, GridND

        valid = isinstance(domain, (Grid1D, GridND)) and domain.dim == self.spatial_dim
        axes = domain.axes if isinstance(domain, GridND) else (domain,)
        if valid:
            valid = all(
                np.allclose(
                    np.diff(np.asarray(axis.nodes)),
                    np.diff(np.asarray(axis.nodes))[0],
                    rtol=1e-7,
                    atol=1e-12,
                )
                if axis.num_nodes > 1
                else True
                for axis in axes
            )
        if not valid:
            raise ValueError(
                f"{type(self).__name__} requires a uniform {self.spatial_dim}-D grid; domain {name!r} ({type(domain).__name__}) does not satisfy it"
            )

    def __call__(self, u):
        """Map a scalar field or channel-last field, preserving spatial axes.

        A scalar field omits the channel axis when in_channels=1. Explicit
        channel arrays have a trailing in_channels axis; leading batch axes
        are supported when the channel axis is present.
        """
        u = jnp.asarray(u)
        scalar = self.in_channels == 1 and u.ndim == self.spatial_dim
        if scalar:
            u = u[..., None]
        if u.ndim < self.spatial_dim + 1 or u.shape[-1] != self.in_channels:
            raise ValueError(
                f"{type(self).__name__} expects {self.spatial_dim} spatial axes and {self.in_channels} input channels; got {u.shape}"
            )
        z = u @ self.lifting.weight.T + self.lifting.bias
        for layer in self.layers:
            z = layer(z)
        result = z @ self.projection.weight.T + self.projection.bias
        return result[..., 0] if scalar and self.out_channels == 1 else result

    def learnable_paths(self) -> tuple[str, ...]:
        """Return real-valued lifting, spectral, and projection parameters."""
        return _array_paths(self)


class FNO1d(_FNO):
    """Fourier neural operator on a uniform one-dimensional sampled field.

    Use modes retained frequencies, width hidden channels, and depth spectral
    layers. A scalar input of shape (n,) returns (n,); explicit channels use
    (n, in_channels) and produce (n, out_channels).
    """

    spatial_dim: ClassVar[int] = 1


class FNO2d(_FNO):
    """Fourier neural operator on a uniform two-dimensional sampled field.

    modes is one integer or a pair of retained frequency counts. A scalar
    input of shape (nx, ny) returns that shape; explicit channel fields use
    (nx, ny, in_channels) and produce (nx, ny, out_channels).
    """

    spatial_dim: ClassVar[int] = 2
