"""Neural approximators with convexity, monotonicity, and output structure."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from .coordinate_nets import _activation, _array_paths, _coordinates


class ICNN(eqx.Module):
    """Scalar input-convex network with unrestricted affine input connections.

    Hidden-to-hidden and hidden-to-output weights are softplus transforms of
    unconstrained fitted arrays. Convex, nondecreasing hidden activations then
    preserve convexity in every input. ``activation`` is softplus or relu.
    Inputs may be a vector or a batch with the input dimension last.
    """

    input_layers: tuple
    hidden_weights: tuple
    output_weights: jax.Array
    output_input: eqx.nn.Linear
    in_size: int = eqx.field(static=True)
    activation: str = eqx.field(static=True)

    def __init__(self, in_size, *, width, depth, key, activation="softplus"):
        """Initialize an input-convex network of positive width and depth."""
        if min(in_size, width, depth) < 1:
            raise ValueError("ICNN in_size, width, and depth must be positive integers")
        if activation not in ("softplus", "relu"):
            raise ValueError(
                "ICNN activation must be convex and nondecreasing: choose 'softplus' or 'relu'"
            )
        keys = jax.random.split(key, 2 * depth + 1)
        self.in_size, self.activation = in_size, activation
        self.input_layers = tuple(eqx.nn.Linear(in_size, width, key=keys[i]) for i in range(depth))
        self.hidden_weights = tuple(
            -2 + 0.1 * jax.random.normal(keys[depth + i], (width, width)) for i in range(depth - 1)
        )
        self.output_weights = -2 + 0.1 * jax.random.normal(keys[-2], (width,))
        self.output_input = eqx.nn.Linear(in_size, 1, key=keys[-1])

    def __call__(self, x):
        """Evaluate the convex scalar function over any leading batch axes."""
        x = _coordinates(x, self.in_size, None)
        activate = _activation(self.activation)
        layer = self.input_layers[0]
        z = activate(x @ layer.weight.T + layer.bias)
        for layer, weights in zip(self.input_layers[1:], self.hidden_weights):
            z = activate(z @ jax.nn.softplus(weights).T + x @ layer.weight.T + layer.bias)
        return (
            z @ jax.nn.softplus(self.output_weights)
            + (x @ self.output_input.weight.T + self.output_input.bias)[..., 0]
        )

    def learnable_paths(self) -> tuple[str, ...]:
        """Return fitted affine leaves and unconstrained positive-weight parameters."""
        return _array_paths(self)


class MonotoneMLP(eqx.Module):
    """Scalar MLP with a declared derivative sign for each input coordinate.

    ``monotone_in`` contains +1 for increasing, -1 for decreasing, and 0 for
    unconstrained inputs; the default is +1 everywhere. Constrained input
    weights have the chosen sign and every later weight is nonnegative,
    all through softplus reparameterization. Activations are monotone.
    """

    mlp: eqx.nn.MLP
    in_size: int = eqx.field(static=True)
    monotone_in: tuple = eqx.field(static=True)
    activation: str = eqx.field(static=True)

    def __init__(self, in_size, *, width, depth, key, monotone_in=None, activation="tanh"):
        """Initialize the sign-constrained weights and free biases."""
        if min(in_size, width) < 1 or depth < 0:
            raise ValueError("MonotoneMLP sizes must be positive and depth nonnegative")
        signs = (1,) * in_size if monotone_in is None else tuple(monotone_in)
        if len(signs) != in_size or any(sign not in (-1, 0, 1) for sign in signs):
            raise ValueError(f"monotone_in must have {in_size} entries, each -1, 0, or +1")
        if activation not in ("tanh", "softplus", "relu", "sigmoid"):
            raise ValueError(
                "MonotoneMLP activation must be 'tanh', 'softplus', 'relu', or 'sigmoid'"
            )
        self.in_size, self.monotone_in, self.activation = in_size, signs, activation
        self.mlp = eqx.nn.MLP(in_size, 1, width, depth, key=key)

    def __call__(self, x):
        """Evaluate a vector or batch while preserving the declared signs."""
        z = _coordinates(x, self.in_size, None)
        signs = jnp.asarray(self.monotone_in)
        for index, layer in enumerate(self.mlp.layers):
            weights = jax.nn.softplus(layer.weight)
            if index == 0:
                weights = jnp.where(signs == 0, layer.weight, weights * signs)
            z = z @ weights.T + layer.bias
            if index < len(self.mlp.layers) - 1:
                z = _activation(self.activation)(z)
        return z[..., 0]

    def learnable_paths(self) -> tuple[str, ...]:
        """Return every fitted raw weight and bias path."""
        return tuple("mlp." + path for path in _array_paths(self.mlp))


class Transform(eqx.Module):
    """Wrap a module with a named output transform or a callable.

    exp, softplus, abs, and square act on the output. even and odd average
    the module at the first positional argument and its negative, leaving
    any further arguments fixed. An inner learnable_paths() declaration is
    preserved with a module. prefix; otherwise every array leaf is fitted.
    """

    module: Any
    fn: Any = eqx.field(static=True)

    def __init__(self, module, fn):
        """Keep the inner module's fitted leaves and validate the transform."""
        if not callable(module):
            raise TypeError("Transform module must be callable")
        if not callable(fn) and fn not in ("exp", "softplus", "abs", "square", "even", "odd"):
            raise ValueError(
                "Transform fn must be callable or 'exp', 'softplus', 'abs', 'square', 'even', or 'odd'"
            )
        self.module, self.fn = module, fn

    def __call__(self, *inputs):
        """Apply the output transform or parity projection."""
        raw = self.module(*inputs)
        if callable(self.fn):
            return self.fn(raw)
        if self.fn in ("even", "odd"):
            reflected = self.module(-jnp.asarray(inputs[0]), *inputs[1:])
            return 0.5 * (raw + reflected if self.fn == "even" else raw - reflected)
        return {"exp": jnp.exp, "softplus": jax.nn.softplus, "abs": jnp.abs, "square": jnp.square}[
            self.fn
        ](raw)

    def learnable_paths(self) -> tuple[str, ...]:
        """Prefix the inner module's fitted paths, preserving any narrowing."""
        declare = getattr(self.module, "learnable_paths", None)
        paths = tuple(declare()) if callable(declare) else _array_paths(self.module)
        return tuple("module." + path for path in paths)

    def _validate_domain(self, domain, *, name):
        """Preserve an inner operator's domain requirements when wrapped."""
        validate = getattr(self.module, "_validate_domain", None)
        if callable(validate):
            validate(domain, name=name)
