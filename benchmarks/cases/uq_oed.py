"""Prediction, information, Laplace, and design-gradient kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from benchmarks._core import Case, Workload

_ALL = ("pr-cpu", "nightly-cpu", "nightly-gpu", "release")


def _oed_gradient() -> Workload:
    theta = jnp.asarray([0.3, 0.7, 1.1])
    design = jnp.asarray([0.2, 0.5, 0.9, 1.3])
    precision = jnp.eye(3) * 0.25

    def prediction(parameters, experiment):
        return jnp.exp(-experiment[:, None] * parameters[None, :])

    def criterion(experiment, parameters):
        sensitivities = jax.jacfwd(prediction, argnums=0)(parameters, experiment)
        fisher = jnp.einsum("tij,tik->jk", sensitivities, sensitivities) + precision
        return jnp.linalg.slogdet(fisher)[1]

    function = jax.grad(criterion, argnums=0)

    def check(value):
        finite = bool(jnp.all(jnp.isfinite(value)))
        return finite, 0.0 if finite else float("inf"), 1e-10

    return Workload(function, (design, theta), check)


def _observed_hessian() -> Workload:
    theta = jnp.asarray([0.3, 0.7, 1.1])
    design = jnp.asarray([0.2, 0.5, 0.9, 1.3])
    observed = jnp.exp(-design[:, None] * jnp.asarray([0.4, 0.8, 1.0])[None, :])

    def nll(parameters, data, experiment):
        residual = jnp.exp(-experiment[:, None] * parameters[None, :]) - data
        return 0.5 * jnp.sum(residual**2) + 0.05 * jnp.sum(parameters**2)

    function = jax.jacfwd(jax.jacfwd(nll, argnums=0), argnums=0)

    def check(value):
        symmetry = float(jnp.max(jnp.abs(value - value.T)))
        finite = bool(jnp.all(jnp.isfinite(value)))
        return finite and symmetry <= 1e-12, symmetry, 1e-12

    return Workload(function, (theta, observed, design), check)


def cases() -> tuple[Case, ...]:
    """Return portable higher-order UQ qualification cases."""
    common = {
        "suites": _ALL,
        "solver": "analytical",
        "backend": "xla-dense",
        "problem": "local-exponential-design",
        "dimensions": {"states": 12, "parameters": 3},
        "sparsity": {"nnz": 9, "density": 1.0},
        "batch_width": 4,
        "direction_count": 3,
    }
    return (
        Case(
            "uq-oed-reverse-over-forward",
            "1",
            build=_oed_gradient,
            transform="grad-jacfwd",
            **common,
        ),
        Case(
            "uq-observed-forward-over-forward",
            "1",
            build=_observed_hessian,
            transform="jacfwd-jacfwd",
            **common,
        ),
    )
