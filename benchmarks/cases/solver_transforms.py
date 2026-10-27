"""Analytical and solver-backed transform cases."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._core import Case, Workload

_ALL = ("pr-cpu", "nightly-cpu", "nightly-gpu", "release")
_REFERENCE = ("nightly-cpu", "nightly-gpu", "release")


def _analytical_jacfwd() -> Workload:
    weights = jnp.linspace(0.2, 0.9, 8)
    parameters = jnp.linspace(0.03, 0.1, 8)
    observations = jnp.asarray([1.7])
    design = jnp.asarray([1.2])

    def residual(theta, data, experiment):
        rate = jnp.dot(theta, weights) * experiment[0]
        return jnp.asarray([jnp.exp(-rate * 0.05) - data[0]])

    function = jax.jacfwd(residual, argnums=0)

    def check(value):
        rate = jnp.dot(parameters, weights) * design[0]
        expected = (-0.05 * design[0] * jnp.exp(-rate * 0.05) * weights)[None, :]
        error = float(jnp.max(jnp.abs(value - expected)))
        return error <= 1e-11, error, 1e-11

    return Workload(function, (parameters, observations, design), check)


def _analytical_vmap() -> Workload:
    width = 16
    weights = jnp.linspace(0.2, 0.9, 8)
    base = jnp.linspace(0.03, 0.1, 8)
    parameters = base[None, :] + jnp.linspace(0.0, 0.07, width)[:, None]
    observations = jnp.asarray([1.7])
    design = jnp.asarray([1.2])

    def residual(theta, data, experiment):
        rate = jnp.dot(theta, weights) * experiment[0]
        return jnp.exp(-rate * 0.05) - data[0]

    function = jax.vmap(residual, in_axes=(0, None, None))

    def check(value):
        expected = jnp.exp(-(parameters @ weights) * design[0] * 0.05) - observations[0]
        error = float(jnp.max(jnp.abs(value - expected)))
        return error <= 1e-12, error, 1e-12

    return Workload(function, (parameters, observations, design), check)


def _solver_transform(engine_name: str, transform: str, width: int) -> Workload:
    import diff_soup as ds

    engine = getattr(ds, engine_name)
    weights = jnp.linspace(0.2, 0.9, width)
    base = jnp.linspace(0.03, 0.1, width)
    problem = ds.MassProblem(
        rhs=lambda _t, y, rate, _control: -rate * y,
        mass=ds.MassOperator.identity(2),
        structure=ds.DAEStructure(differential_vars=np.ones((2,), dtype=bool)),
    )
    linear = "cudss" if jax.default_backend() == "gpu" else "dense"
    solver = engine.stock(rtol=1e-6, atol=1e-8, linear=linear)
    # Reverse mode replays checkpointed segments; the other transforms leave the adjoint unset.
    reverse = {"adjoint": ds.CheckpointAdjoint()} if transform == "grad" else {}

    def endpoint(parameters):
        rate = jnp.dot(parameters, weights)
        return jnp.sum(
            ds.pdesolve(
                problem,
                solver,
                (0.0, 0.05),
                jnp.asarray([1.0, 2.0]),
                theta=rate,
                saveat=ds.SaveAt(t1=True),
                consistent_ic=False,
                max_steps=100,
                **reverse,
            ).ys[-1]
        )

    if transform == "jacfwd":
        function = jax.jacfwd(endpoint)
        arguments = (base,)

        def check(value):
            rate = jnp.dot(base, weights)
            expected = -0.05 * 3.0 * jnp.exp(-rate * 0.05) * weights
            error = float(jnp.max(jnp.abs(value - expected)))
            return error <= 2e-3, error, 2e-3

    elif transform == "grad":
        # Reverse mode through the engine: the reverse sweep solves the transposed systems on
        # the forward factors (one transposed solve per corrector step on cuDSS).
        function = jax.grad(endpoint)
        arguments = (base,)

        def check(value):
            rate = jnp.dot(base, weights)
            expected = -0.05 * 3.0 * jnp.exp(-rate * 0.05) * weights
            error = float(jnp.max(jnp.abs(value - expected)))
            return error <= 2e-3, error, 2e-3

    elif transform == "scalar-jvp":
        basis = jnp.eye(width, dtype=base.dtype)

        def function(parameters):
            return jnp.stack(
                [jax.jvp(endpoint, (parameters,), (basis[index],))[1] for index in range(width)]
            )

        arguments = (base,)

        def check(value):
            rate = jnp.dot(base, weights)
            expected = -0.05 * 3.0 * jnp.exp(-rate * 0.05) * weights
            error = float(jnp.max(jnp.abs(value - expected)))
            return error <= 2e-3, error, 2e-3

    else:
        parameters = base[None, :] + jnp.linspace(0.0, 0.07, width)[:, None]
        if transform == "vmap":
            function = jax.vmap(endpoint)
        else:

            def function(batch):
                return jnp.stack([endpoint(batch[index]) for index in range(width)])

        arguments = (parameters,)

        def check(value):
            expected = 3.0 * jnp.exp(-(parameters @ weights) * 0.05)
            error = float(jnp.max(jnp.abs(value - expected)))
            return error <= 2e-3, error, 2e-3

    return Workload(function, arguments, check)


def _root_pullback(width: int) -> Workload:
    """The gradient of a mass-matrix DAE's endpoint through its consistent initialization.

    ``y0' = -rate y0`` with the algebraic row ``0 = y1 - rate y0`` and the inconsistent start
    ``(1, 0)``: the consistent initialization is a root solve whose pullback carries the
    parameter's effect on the start, and the reverse sweep then solves the transposed systems of
    the DAE integrator. The endpoint ``y0 + y1`` at ``T`` is ``(1 + rate) exp(-rate T)``.
    """
    import diff_soup as ds

    weights = jnp.linspace(0.2, 0.9, width)
    base = jnp.linspace(0.03, 0.1, width)
    horizon = 0.05
    problem = ds.MassProblem(
        rhs=lambda _t, y, rate, _control: jnp.stack([-rate * y[0], y[1] - rate * y[0]]),
        mass=ds.MassOperator.sparse(
            [0],
            [0],
            lambda _t, y, _rate, _control: jnp.ones((1,), y.dtype),
            n=2,
            dependencies=ds.Dependencies(time=False, state=False, parameters=False, control=False),
        ),
        structure=ds.DAEStructure(differential_vars=np.array([True, False])),
    )
    linear = "cudss" if jax.default_backend() == "gpu" else "dense"
    solver = ds.JIDA.stock(rtol=1e-6, atol=1e-8, linear=linear)

    def endpoint(parameters):
        rate = jnp.dot(parameters, weights)
        return jnp.sum(
            ds.pdesolve(
                problem,
                solver,
                (0.0, horizon),
                jnp.asarray([1.0, 0.0]),
                theta=rate,
                saveat=ds.SaveAt(t1=True),
                consistent_ic=True,
                max_steps=100,
                adjoint=ds.CheckpointAdjoint(),
            ).ys[-1]
        )

    def check(value):
        rate = jnp.dot(base, weights)
        expected = jnp.exp(-rate * horizon) * (1.0 - horizon * (1.0 + rate)) * weights
        error = float(jnp.max(jnp.abs(value - expected)))
        return error <= 2e-3, error, 2e-3

    return Workload(jax.grad(endpoint), (base,), check)


def cases() -> tuple[Case, ...]:
    """Return solver transform cases, including the required width matrix."""
    result = [
        Case(
            "analytical-decay-jacfwd-8",
            "1",
            _ALL,
            _analytical_jacfwd,
            "analytical",
            "xla-dense",
            "jacfwd",
            "decay-ode",
            {"states": 1, "parameters": 8},
            {"nnz": 1, "density": 1.0},
            1,
            8,
        ),
        Case(
            "analytical-decay-vmap-16",
            "1",
            _ALL,
            _analytical_vmap,
            "analytical",
            "xla-dense",
            "vmap",
            "decay-ode",
            {"states": 1, "parameters": 8},
            {"nnz": 1, "density": 1.0},
            16,
            0,
        ),
    ]
    for engine_name in ("JNDF", "JBDF", "JIDA"):
        for width in (1, 2, 4, 8, 16):
            for transform in ("jacfwd", "vmap", "grad"):
                result.append(
                    Case(
                        f"{engine_name.lower()}-decay-{transform}-{width}",
                        "1",
                        _REFERENCE,
                        partial(_solver_transform, engine_name, transform, width),
                        engine_name,
                        "cudss-or-dense",
                        transform,
                        "decay-ode",
                        {"states": 2, "parameters": width},
                        {"nnz": 2, "density": 0.5},
                        width if transform == "vmap" else 1,
                        width if transform == "jacfwd" else 0,
                        single_lane_build=partial(_solver_transform, engine_name, transform, 1),
                    )
                )
    for width in (1, 4, 16):
        result.append(
            Case(
                f"jida-dae-root-grad-{width}",
                "1",
                _REFERENCE,
                partial(_root_pullback, width),
                "JIDA",
                "cudss-or-dense",
                "grad",
                "decay-dae-consistent-ic",
                {"states": 2, "parameters": width},
                {"nnz": 3, "density": 0.75},
                1,
                0,
            )
        )
    result.extend(
        (
            Case(
                "jndf-decay-scalar-jvp-8",
                "1",
                _REFERENCE,
                partial(_solver_transform, "JNDF", "scalar-jvp", 8),
                "JNDF",
                "cudss-or-dense",
                "sequential-scalar-jvp",
                "decay-ode",
                {"states": 2, "parameters": 8},
                {"nnz": 2, "density": 0.5},
                1,
                8,
                single_lane_build=partial(_solver_transform, "JNDF", "scalar-jvp", 1),
            ),
            Case(
                "jndf-decay-scalar-vmap-16",
                "1",
                _REFERENCE,
                partial(_solver_transform, "JNDF", "scalar-vmap", 16),
                "JNDF",
                "cudss-or-dense",
                "sequential-scalar-dispatch",
                "decay-ode",
                {"states": 2, "parameters": 16},
                {"nnz": 2, "density": 0.5},
                16,
                0,
                single_lane_build=partial(_solver_transform, "JNDF", "scalar-vmap", 1),
            ),
        )
    )
    return tuple(result)
