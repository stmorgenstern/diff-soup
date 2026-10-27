"""The small stiff mass-matrix problem the second-order AD contracts share, its loss, and the
composable full-precision reference (public imports only)."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

import diff_soup as ds

Y0, THETA, SPAN = jnp.asarray([1.0, 0.5]), jnp.asarray([1.3, 0.7]), (0.0, 0.3)
#: Solver settings (requested tolerances, not asserted bounds).
TIGHT = {"rtol": 1e-10, "atol": 1e-12}

PROBLEM = ds.MassProblem(
    rhs=lambda t, y, theta, control: jnp.stack(
        [-theta[0] * y[0] * y[1] + jnp.sin(t), y[0] - y[1] ** 2 + theta[1]]
    ),
    mass=ds.MassOperator.diagonal(
        lambda t, y, theta, control: jnp.asarray([1.0, 2.0]),
        n=2,
        dependencies=ds.Dependencies(time=False, state=False, parameters=False, control=False),
    ),
    structure=ds.DAEStructure(differential_vars=np.array([True, True])),
)


def loss(adjoint=None, linear="dense", precision=None, engine=ds.JNDF):
    """The squared end state of the solve as a function of theta."""
    options = {} if precision is None else {"precision": precision}
    solver = engine.stock(linear=linear, **TIGHT, **options)

    def squared_end(theta):
        solution = ds.pdesolve(
            PROBLEM, solver, SPAN, Y0, theta=theta, adjoint=adjoint, max_steps=2000
        )
        return jnp.sum(solution.ys[-1] ** 2)

    return squared_end


def gradient_and_hessian(gradient, theta=THETA) -> tuple[np.ndarray, np.ndarray]:
    """``gradient`` at ``theta`` and the forward pushes over it along each unit direction (the
    columns of ``jax.hessian``), one compiled push evaluated once per direction."""
    push = jax.jit(lambda point, direction: jax.jvp(gradient, (point,), (direction,)))
    columns = [push(theta, direction) for direction in jnp.eye(theta.shape[0], dtype=theta.dtype)]
    hessian = np.stack([np.asarray(tangent) for _, tangent in columns], axis=1)
    return np.asarray(columns[0][0]), hessian


@functools.cache
def composable_reference() -> tuple[np.ndarray, np.ndarray]:
    """The reverse-over-forward Hessian and the forward gradient at ``THETA`` under
    ``ForwardAdjoint(composable=True)``, the full-precision reference: one compiled reverse pass
    over a forward push, evaluated once per unit direction. Cached at the Python level, so the
    modules that read it compute it once per session."""
    squared_end = loss(ds.ForwardAdjoint(composable=True))
    row = jax.jit(
        jax.value_and_grad(lambda theta, direction: jax.jvp(squared_end, (theta,), (direction,))[1])
    )
    rows = [row(THETA, direction) for direction in jnp.eye(THETA.shape[0], dtype=THETA.dtype)]
    hessian = np.stack([np.asarray(gradient) for _, gradient in rows])
    return hessian, np.asarray([float(value) for value, _ in rows])
