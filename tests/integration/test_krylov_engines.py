"""Engines with an iterative linear backend: GMRES in the corrector, preconditioned by the
factor of the last setup, and in the reverse sweep's cotangent solves; the cuDSS reverse
sweep's held preconditioner is owed on a CPU host."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.domains import cartesian_grid_1d
from diff_soup.linear import Dense, Krylov, SuperLU


def _dae1d(n, scale):
    """``(problem, state, theta)``: a periodic 1D reaction-diffusion DAE, a differential field
    ``c`` consumed at rate ``k`` and an algebraic equilibrium field ``w`` of size ``scale``."""
    spec = ds.problem(f"krylov_dae1d_{n}_{scale}").domain(
        "rod", cartesian_grid_1d(n, 0.0, 1.0, method="fvm")
    )
    rate = spec.param("k", value=1.0, learnable=True)
    spec.field("c", mass="differential", scale=1.0)
    spec.field("w", mass="algebraic", scale=1.0)
    spec.equation(writes="c", reads=("c", "w"))(
        lambda q: q.ops.laplacian(q.c) - q.p.k * (q.w / scale) * q.c
    )
    spec.equation(writes="w", reads=("c", "w"))(
        lambda q: q.w - 1e-3 * q.ops.laplacian(q.w) - scale * q.c / (0.5 + q.c)
    )
    spec.bc.periodic("c")
    spec.bc.periodic("w")
    c0 = 0.5 + 0.25 * np.sin(2.0 * np.pi * (np.arange(n) + 0.5) / n)
    state = spec.assemble_state({"c": jnp.asarray(c0), "w": jnp.asarray(scale * c0 / (0.5 + c0))})
    problem = spec.build()
    return problem, jnp.asarray(state), ds.default_theta(problem, {rate: 1.0})


def _route(linear, engine="JNDF", n=24):
    """The value, solve report and parameter gradient of the mean of ``c`` at ``t = 0.5``."""
    problem, state, theta = _dae1d(n, 1.0)

    def loss(th):
        solver = getattr(ds, engine).stock(
            linear=linear, adjoint=ds.CheckpointAdjoint(), **{"rtol": 1e-5, "atol": 1e-7}
        )
        solution = ds.pdesolve(
            problem,
            solver,
            (0.0, 0.5),
            state,
            theta=th,
            saveat=ds.SaveAt(t1=True),
            consistent_ic=True,
            max_steps=400,
        )
        report = (solution.ok, solution.stats.accepts, solution.stats.newton_failures)
        return jnp.mean(solution.ys[-1][:n]), report

    (value, report), gradient = jax.value_and_grad(loss, has_aux=True)(theta)
    leaves = jax.tree_util.tree_leaves(gradient)
    return value, report, np.concatenate([np.ravel(np.asarray(leaf)) for leaf in leaves])


# measured 6.3e-8 (value) and 1.3e-6 (gradient) relative between the routes at rtol 1e-5,
# 55 and 54 steps, CPU VM 2026-10-22,
# pytest tests/integration/test_krylov_engines.py::test_a_host_preconditioned_krylov_route_gives_the_exact_solution_and_gradient
_ROUTE = 1e-5


@pytest.mark.requires_superlu
def test_a_host_preconditioned_krylov_route_gives_the_exact_solution_and_gradient():
    """GMRES preconditioned by a SuperLU_MT factor, in the corrector and in the cotangent
    solves of the checkpointed reverse sweep, gives the exact SuperLU_MT route's value and
    gradient to the integrator's tolerance without a Newton failure."""
    value_e, report_e, gradient_e = _route(SuperLU())
    value_k, report_k, gradient_k = _route(Krylov(SuperLU()))
    assert bool(report_e[0]) and bool(report_k[0]) and int(report_k[2]) == 0
    np.testing.assert_allclose(value_k, value_e, rtol=_ROUTE)
    assert np.linalg.norm(gradient_k - gradient_e) <= _ROUTE * np.linalg.norm(gradient_e)


# measured 35 steps on both routes and an end-state difference of 1.08 times the relative
# tolerance (1.1e-5 at rtol 1e-5), CPU VM 2026-10-22,
# pytest tests/integration/test_krylov_engines.py::test_the_krylov_corrector_converges_under_non_uniform_error_weights
_STEPS, _END = 1.2, 10.0


def test_the_krylov_corrector_converges_under_non_uniform_error_weights():
    """With an algebraic field six decades smaller than the differential one (error weights
    varying about seventy-fold) the Krylov corrector over a Dense preconditioner never fails
    a Newton iteration, takes at most 1.2 times the exact route's steps, and ends within ten
    relative tolerances of it."""
    problem, state, theta = _dae1d(32, 1e-6)
    rtol = 1e-5

    def forward(linear):
        solver = ds.JNDF.stock(linear=linear, **{"rtol": rtol, "atol": 1e-2 * rtol})
        solution = ds.pdesolve(
            problem,
            solver,
            (0.0, 0.2),
            state,
            theta=theta,
            saveat=ds.SaveAt(t1=True),
            consistent_ic=True,
            max_steps=1000,
        )
        return solution.ys[-1], solution.ok, solution.stats.accepts, solution.stats.newton_failures

    end_e, ok_e, steps_e, _ = jax.jit(lambda: forward(Dense()))()
    end_k, ok_k, steps_k, failures = jax.jit(lambda: forward(Krylov(Dense())))()
    assert bool(ok_e) and bool(ok_k) and int(failures) == 0
    assert int(steps_k) <= _STEPS * int(steps_e)
    assert float(jnp.max(jnp.abs(end_k - end_e))) <= _END * rtol * float(jnp.max(jnp.abs(end_e)))


# Gradient difference relative to the exact route's at rtol 1e-5: measured 1.3e-6 (JNDF, 55
# and 54 steps), 1.28e-5 (JBDF, 64 and 59 steps) and 4.1e-6 (JIDA, 65 and 64 steps), values within
# 6.2e-7, the same to three digits over SuperLU_MT and Dense on CPU; the JBDF difference falls to
# 3.2e-8 at rtol 1e-6 (85 steps both), RTX 4070 Ti 2026-10-23,
# pytest tests/integration/test_krylov_engines.py::test_a_cudss_preconditioned_reverse_sweep_gives_the_exact_gradient
_CUDSS_GRADIENT = {"JNDF": 1e-5, "JBDF": 1e-4, "JIDA": 1e-5}


@pytest.mark.requires_cudss
@pytest.mark.parametrize("engine", ["JNDF", "JBDF", "JIDA"])
def test_a_cudss_preconditioned_reverse_sweep_gives_the_exact_gradient(engine):
    """Over cuDSS the reverse sweep preconditions every cotangent solve with the factor of
    ``A^T`` held in shared storage: the value and gradient equal the exact cuDSS route's to
    the integrator's tolerance without a Newton failure."""
    from diff_soup.linear import CuDSS

    value_e, report_e, gradient_e = _route("cudss", engine)
    value_k, report_k, gradient_k = _route(Krylov(CuDSS()), engine)
    assert bool(report_e[0]) and bool(report_k[0]) and int(report_k[2]) == 0
    np.testing.assert_allclose(value_k, value_e, rtol=_ROUTE)
    bound = _CUDSS_GRADIENT[engine] * np.linalg.norm(gradient_e)
    assert np.linalg.norm(gradient_k - gradient_e) <= bound
