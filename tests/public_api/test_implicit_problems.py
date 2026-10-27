"""Implicit problems and root solves: the public residual of mass-form and fully implicit
problems, consistent initialization, and the algebraic projection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from tests.support.problems import SparseChainDAE, reaction_diffusion_recipe

_CONSTANT = ds.Dependencies(time=False, state=False, parameters=False, control=False)
_OPTIONS = ds.AlgebraicProjectionOptions(
    **{"rtol": 1e-10, "atol": 1e-12}, max_steps=25, linear_solver="dense"
)
# measured 0.0 on the residual, projection and consistent-start comparisons below, CPU VM
# 2026-10-22, pytest tests/public_api/test_implicit_problems.py
_EXACT = 1e-12
# measured 3.5e-8 relative between JIDA at rtol 1e-8 and a 400-step RK4 reference, same run
_ROUTE_BOUND = 1e-6


def test_a_mass_form_problem_presents_its_residual_and_refuses_algebraic_derivative_support():
    """``D ydot - f`` with a parameter-weighted derivative coupling into an algebraic row is the
    residual, never reads the algebraic slope, and differentiates through ``D``'s parameters;
    derivative support in an algebraic column is refused; a recipe's ``field_slice`` names
    its field's rows."""
    coupled = ds.MassOperator.block_sparse(
        blocks=(((0, 1), (0, 1), jnp.eye(2)),),
        couplings=((2, (0, 1), lambda t, y, theta, control: theta["w"]),),
        n=3,
        dependencies=ds.Dependencies(time=False, state=False, parameters=True, control=False),
    )
    problem = ds.MassProblem(
        rhs=lambda t, y, theta, control: jnp.stack([-y[0], -y[1], y[2]]),
        mass=coupled,
        structure=ds.DAEStructure(differential_vars=np.array([True, True, False])),
    )
    theta, y, ydot = (
        {"w": jnp.asarray([0.25, 0.75])},
        jnp.asarray([1.0, 2.0, 5.0]),
        jnp.asarray([0.1, 0.2, 99.0]),
    )
    residual = problem.residual(0.0, y, ydot, theta, None)
    np.testing.assert_allclose(residual, [1.1, 2.2, 0.25 * 0.1 + 0.75 * 0.2 - 5.0], atol=_EXACT)
    gradient = jax.grad(lambda th: problem.residual(0.0, y, ydot, th, None)[2])(theta)
    np.testing.assert_allclose(gradient["w"], ydot[:2], atol=_EXACT)
    with pytest.raises(ValueError):
        ds.MassProblem(
            rhs=lambda t, y, theta, control: -y,
            mass=ds.MassOperator.diagonal(jnp.ones(2)),
            structure=ds.DAEStructure(differential_vars=np.array([True, False])),
        )
    assert reaction_diffusion_recipe("sliced", n=5).build().field_slice("u") == slice(0, 5)


def test_a_fully_implicit_residual_solves_initializes_and_reports_an_impossible_start():
    """A residual nonlinear in the slope (no mass form) integrates through JIDA to the
    reference, its ``y_fixed_ydot`` initialization zeroes the residual, and a residual with
    no root reports INCONSISTENT_IC."""
    sparsity = ds.ResidualLinearization(sparsity=ds.ResidualSparsity(dy="dense", dydot="dense"))
    problem = ds.ImplicitProblem(
        residual=lambda t, y, ydot, theta, control: ydot + theta * y + 0.1 * ydot**3,
        structure=ds.DAEStructure(differential_vars=np.array([True])),
        linearization=sparsity,
    )
    k = jnp.asarray(0.9)

    def slope(y):  # the slope g solving g + k y + 0.1 g^3 = 0, by Newton
        return jax.lax.fori_loop(
            0, 30, lambda _, g: g - (g + k * y + 0.1 * g**3) / (1.0 + 0.3 * g**2), -k * y
        )

    def rk4(_, y, dt=0.25 / 400):
        k1 = slope(y)
        k2 = slope(y + 0.5 * dt * k1)
        k3 = slope(y + 0.5 * dt * k2)
        return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + slope(y + dt * k3))

    def solve(ic_mode):
        solver = ds.JIDA.stock(linear="dense", ic_mode=ic_mode, **{"rtol": 1e-8, "atol": 1e-10})
        saveat = ds.SaveAt(t0=True, t1=True)
        return ds.pdesolve(problem, solver, (0.0, 0.25), jnp.asarray([2.0]), theta=k, saveat=saveat)

    end = solve("ya_ydp").ys[-1, 0]
    np.testing.assert_allclose(end, jax.lax.fori_loop(0, 400, rk4, 2.0), rtol=_ROUTE_BOUND)
    start = solve("y_fixed_ydot").ys[0]
    np.testing.assert_allclose(
        problem.residual(0.0, start, jnp.zeros(1), k, None), 0.0, atol=_EXACT
    )
    impossible = ds.ImplicitProblem(
        residual=lambda t, y, ydot, theta, control: jnp.ones_like(y),
        structure=ds.DAEStructure(differential_vars=np.array([False])),
        linearization=sparsity,
    )
    failed = ds.pdesolve(
        impossible, ds.JIDA.stock(linear="dense", ic_mode="ya_ydp_euler"), (0.0, 0.1), jnp.zeros(1)
    )
    assert failed.result == ds.RESULT.INCONSISTENT_IC


def test_a_solve_starts_from_the_consistent_projection_of_an_off_manifold_state():
    """A sparse DAE started with its algebraic chain and global at zero saves, at ``t0``, the
    exact consistent state: differential rows held, algebraic rows solved."""
    case = SparseChainDAE()
    solution = ds.pdesolve(
        case.problem,
        ds.JIDA.stock(linear="dense"),
        (0.0, 0.01),
        case.initial_state(),
        theta=jnp.asarray(0.4),
        saveat=ds.SaveAt(t0=True),
        consistent_ic=True,
    )
    start = jnp.concatenate([solution.ys[0], solution.lambdas[0]])
    assert int(solution.stats.ic_solves) == 1
    np.testing.assert_allclose(start, case.analytical(0.0, 0.4), rtol=0.0, atol=_EXACT)


def _quadratic():
    """``x0' = -x0`` and the algebraic row ``0 = x1^2 - 4``."""
    return ds.MassProblem(
        rhs=lambda t, y, theta, control: jnp.stack([-y[0], y[1] ** 2 - 4.0]),
        mass=ds.MassOperator.sparse(
            [0],
            [0],
            lambda t, y, theta, control: jnp.ones((1,), y.dtype),
            n=2,
            dependencies=_CONSTANT,
        ),
        structure=ds.DAEStructure(differential_vars=np.array([True, False])),
    )


def test_project_algebraic_solves_the_algebraic_rows_and_reports_a_typed_status():
    """The projection holds the differential row, solves ``x1 = 2`` from its seed, reports
    SINGULAR_JACOBIAN and NONFINITE per vmap lane, raises the typed error from
    ``require_success`` on the host (refusing a traced result), and asks for ``state_dot``
    when a derivative feeds an algebraic row."""
    problem, state = _quadratic(), jnp.asarray([1.0, 0.0])

    def project(seed):
        result = ds.project_algebraic(
            problem, t=0.0, state=state, seed=jnp.stack([0.0, seed]), options=_OPTIONS
        )
        return result.state, result.status

    states, statuses = jax.jit(jax.vmap(project))(jnp.asarray([2.5, 0.0, jnp.nan]))
    assert [int(s) for s in statuses] == [0, 2, 3]
    np.testing.assert_array_equal(states[0, 0], 1.0)
    np.testing.assert_allclose(states[0, 1], 2.0, rtol=_EXACT)
    capped = ds.AlgebraicProjectionOptions(max_steps=1, linear_solver="dense")
    stalled = ds.project_algebraic(
        problem, t=0.0, state=state, seed=jnp.asarray([0.0, 40.0]), options=capped
    )
    assert int(stalled.status) == int(ds.AlgebraicProjectionStatus.MAX_ITERATIONS)
    with pytest.raises(ds.AlgebraicProjectionError, match="MAX_ITERATIONS"):
        stalled.require_success()
    with pytest.raises(TypeError, match="require_success"):
        jax.jit(
            lambda s: ds.project_algebraic(
                problem, t=0.0, state=state, seed=s, options=_OPTIONS
            ).require_success()
        )(state)
    fed = ds.MassProblem(
        rhs=lambda t, y, theta, control: -y,
        mass=ds.MassOperator.sparse(
            [0, 1],
            [0, 0],
            lambda t, y, theta, control: jnp.asarray([1.0, -1.7], y.dtype),
            n=2,
            dependencies=_CONSTANT,
        ),
        structure=ds.DAEStructure(differential_vars=np.array([True, False])),
    )
    with pytest.raises(ds.ConfigurationError, match="state_dot"):
        ds.project_algebraic(fed, t=0.0, state=state, seed=jnp.zeros(2), options=_OPTIONS)
    result = ds.project_algebraic(
        fed,
        t=0.0,
        state=state,
        state_dot=jnp.asarray([-0.6, 0.0]),
        seed=jnp.zeros(2),
        options=_OPTIONS,
    )
    np.testing.assert_allclose(result.require_success().state[1], 1.7 * -0.6, rtol=_EXACT)


_ROOT = Path(__file__).resolve().parents[2]
# measured 4.3e-8 relative at D = 2.0 and 1.0e-10 at D = 0.5 for each spelling, CPU VM
# 2026-10-23, pytest tests/public_api/test_implicit_problems.py::test_two_mass_problems_whose_rhs_reads_a_rebound_global_solve_differently
_DECAY_BOUND = 1e-6
#: Rebinds a global of its own process between two problem constructions, which the test
#: process may not do, and solves each problem right after building it. Its argument names
#: how the rhs reads the rate: a bare module global, a module's attribute or a class's.
_REBOUND_GLOBAL_WORKER = r"""
import json
import sys
import types

import jax.numpy as jnp
import numpy as np

import diff_soup as ds

D = 0.5
cfg = types.ModuleType("cfg")


class Params:
    D = 0.5


def rhs_bare(t, y, theta, control):
    return -D * y


def rhs_module_attribute(t, y, theta, control):
    return -cfg.D * y


def rhs_class_attribute(t, y, theta, control):
    return -Params.D * y


rhs = {
    "bare": rhs_bare,
    "module_attribute": rhs_module_attribute,
    "class_attribute": rhs_class_attribute,
}[sys.argv[1]]
solver = ds.JIDA.stock(linear="dense", rtol=1e-8, atol=1e-10)
ends = {}
for value in (0.5, 2.0):
    D = cfg.D = Params.D = value
    problem = ds.MassProblem(
        rhs=rhs,
        mass=ds.MassOperator.identity(1),
        structure=ds.DAEStructure(differential_vars=np.array([True])),
    )
    ends[value] = float(ds.pdesolve(problem, solver, (0.0, 1.0), jnp.asarray([1.0])).ys[-1, 0])
print("REPORT " + json.dumps({"rates": list(ends), "ends": list(ends.values())}))
"""


@pytest.mark.parametrize("spelling", ("bare", "module_attribute", "class_attribute"))
def test_two_mass_problems_whose_rhs_reads_a_rebound_global_solve_differently(spelling):
    """Two mass-form problems built from one ``rhs`` that reads the rate as a bare module
    global ``D``, as a module's attribute ``cfg.D`` or as a class's attribute ``Params.D``,
    rebound from 0.5 to 2.0 between the two constructions, each solve ``y' = -D y`` from
    ``y0 = 1`` to their own end state ``exp(-D)``: the second solve does not reuse the first
    one's program."""
    env = {key: value for key, value in os.environ.items() if "proxy" not in key.lower()}
    env.update(
        JAX_PLATFORMS="cpu",
        OMP_NUM_THREADS="1",
        JAX_ENABLE_X64="1",
        JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="1",
    )
    run = subprocess.run(
        [sys.executable, "-c", _REBOUND_GLOBAL_WORKER, spelling],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    line = next(item for item in run.stdout.splitlines() if item.startswith("REPORT "))
    report = json.loads(line.removeprefix("REPORT "))
    np.testing.assert_allclose(
        report["ends"], np.exp(-np.asarray(report["rates"])), rtol=_DECAY_BOUND
    )
