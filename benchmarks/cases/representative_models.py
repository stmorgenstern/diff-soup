"""Representative cohort and replay-shaped kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from benchmarks._core import Case, Workload

_REFERENCE = ("nightly-cpu", "nightly-gpu", "release")


def _owner_cohort() -> Workload:
    shared = jnp.asarray([0.4, 0.6])
    local = jnp.linspace(0.1, 0.8, 16).reshape(8, 2)
    observations = jnp.linspace(0.2, 1.0, 8)

    def child(parameters, child_parameters, target):
        prediction = jnp.dot(parameters, child_parameters)
        return prediction - target

    function = jax.vmap(child, in_axes=(None, 0, 0))

    def check(value):
        expected = local @ shared - observations
        error = float(jnp.max(jnp.abs(value - expected)))
        return error <= 1e-12, error, 1e-12

    return Workload(function, (shared, local, observations), check)


def _reverse_replay() -> Workload:
    controls = jnp.linspace(0.1, 0.4, 32)
    initial = jnp.asarray(1.0)

    def trajectory(theta, values):
        def step(state, control):
            updated = state + 0.01 * (-theta * state + control)
            return updated, updated

        return jax.lax.scan(step, initial, values)[1][-1]

    function = jax.grad(trajectory, argnums=0)

    def check(value):
        finite = bool(jnp.isfinite(value))
        return finite, 0.0 if finite else float("inf"), 1e-10

    return Workload(function, (jnp.asarray(0.7), controls), check)


def cases() -> tuple[Case, ...]:
    """Return owner-aware and recurrence cases."""
    return (
        Case(
            "owner-aware-joint-cohort",
            "1",
            _REFERENCE,
            _owner_cohort,
            "none",
            "xla-dense",
            "vmap",
            "owner-aware-joint",
            {"states": 8, "parameters": 18},
            {"nnz": 16, "density": 1.0},
            8,
            0,
        ),
        Case(
            "reverse-checkpoint-replay",
            "1",
            _REFERENCE,
            _reverse_replay,
            "none",
            "xla-dense",
            "grad-scan-recurrence",
            "short-spm-replay-proxy",
            {"states": 1, "parameters": 1},
            {"nnz": 1, "density": 1.0},
            1,
            1,
        ),
    )
