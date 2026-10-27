"""Linear derivatives: mixed-precision refinement and its derivative, reduced-precision routes
that fall back to a full-precision factor, the options a route analyzes a backend with, the
consumers that need an exact inverse, and the second derivatives and trace-once programs of a
root solve's implicit rules."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.linear import (
    CuDSS,
    Dense,
    FactorContractError,
    RefineReport,
    SolveContext,
    SparseTopology,
    Tolerance,
    analyze,
    factor,
    refine,
    solve,
)
from diff_soup.nonlinear._kkt import LinearPlan, RouteFactor, linear_route

N = 10
# measured at most 9.0e-15 relative: refinement 5.8e-15 (the fp32 solve alone 4.6e-8), its
# tangent and cotangent 9.0e-15, the fallen-back route 0.0, CPU VM 2026-10-22,
# pytest tests/contracts/test_linear_ad.py
_EXACT = 1e-12
# measured 2.7e-13 relative against the closed-form Hessian and 0.0 between the entries'
# gradients, CPU VM 2026-10-22,
# pytest tests/contracts/test_linear_ad.py::test_root_second_derivatives_match_the_closed_form_and_the_entries_agree
_ROOT = 1e-11
FP32_BACKENDS = [
    pytest.param(Dense, id="dense"),
    pytest.param(lambda: CuDSS(), id="cudss", marks=pytest.mark.requires_cudss),
]


def _system(n=N, seed=5, dirichlet=False):
    """``(topology, values, matrix)``: a nonsymmetric, diagonally dominant system."""
    pairs = [(i, j) for i in range(n) for j in (i - 2, i - 1, i, i + 1, i + 3) if 0 <= j < n]
    rows, cols = (np.asarray(axis, np.int32) for axis in zip(*pairs))
    values = np.random.default_rng(seed).uniform(-1.0, 1.0, rows.size) + np.where(
        rows == cols, 5.0, 0.0
    )
    bc = np.asarray([1], np.int32) if dirichlet else None
    matrix = np.zeros((n, n))
    np.add.at(matrix, (rows, cols), values)
    if dirichlet:
        matrix[bc] = 0.0
        matrix[bc, bc] = 1.0
    return (
        SparseTopology(rows=rows, cols=cols, n_dofs=n, bc_indices=bc),
        jnp.asarray(values),
        matrix,
    )


def _relative(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


_B = jnp.asarray(np.cos(np.arange(N) * 0.7) + 0.1 * np.arange(N))
_DB = jnp.asarray(np.sin(np.arange(N)))


@pytest.mark.parametrize("make_backend", FP32_BACKENDS)
def test_refinement_reaches_double_precision_from_a_single_precision_factor_or_stalls(make_backend):
    """Refining an fp32 factor reaches the fp64 solution in a few steps where the plain
    single-precision solve does not, under vmap per lane; a singular system stalls with no
    finite entry; invalid steps or values are refused."""
    backend = make_backend()
    topology, values, matrix = _system(dirichlet=True)
    with SolveContext() as ctx:
        context = None if type(backend).factor_residency == "arrays" else ctx
        a32 = analyze(topology, backend=backend, precision="fp32")
        f = factor(a32, values.astype(jnp.float32), ctx=context)
        x, _, report = refine(f, _B, values=values)
        expected = np.linalg.solve(matrix, np.asarray(_B))
        assert isinstance(report, RefineReport) and x.dtype == jnp.float64
        assert not bool(report.stalled) and 1 <= int(report.steps) <= 5
        assert _relative(x, expected) <= _EXACT < _relative(solve(f, _B)[0], expected)
        lanes = jnp.stack([values, 1.05 * values])
        xs, _, reports = jax.jit(
            jax.vmap(
                lambda v: refine(factor(a32, v.astype(jnp.float32), ctx=context), _B, values=v)
            )
        )(lanes)
        assert reports.steps.shape == (2,) and not np.any(np.asarray(reports.stalled))
        singular = np.arange(1.0, 17.0).reshape(4, 4)
        full = SparseTopology(
            rows=np.repeat(np.arange(4), 4), cols=np.tile(np.arange(4), 4), n_dofs=4
        )
        x, _, report = refine(
            factor(analyze(full, backend=backend), jnp.asarray(singular.ravel()), ctx=context),
            jnp.ones(4),
            values=jnp.asarray(singular.ravel()),
        )
        assert bool(report.stalled) and not np.all(np.isfinite(np.asarray(x)))
        with pytest.raises(FactorContractError):
            refine(f, _B, values=values, max_steps=-1)
        with pytest.raises(FactorContractError):
            refine(f, _B, values=values[:-1])


def test_a_refined_solve_differentiates_as_the_double_precision_solve():
    """The tangent and cotangent of a solve refined from an fp32 factor match the fp64 solve's,
    and its Hessian in the right-hand side is finite."""
    topology, values, _ = _system(dirichlet=True)
    a32, a64 = (
        analyze(topology, backend=Dense(), precision="fp32"),
        analyze(topology, backend=Dense()),
    )

    def refined(v, rhs):
        return refine(factor(a32, v.astype(jnp.float32)), rhs, values=v, max_steps=5)[0]

    def exact(v, rhs):
        return solve(factor(a64, v), rhs)[0]

    dv = 0.1 * values * jnp.asarray(np.random.default_rng(11).uniform(-1, 1, values.shape))
    assert (
        _relative(
            jax.jvp(refined, (values, _B), (dv, _DB))[1], jax.jvp(exact, (values, _B), (dv, _DB))[1]
        )
        <= _EXACT
    )
    for got, want in zip(jax.vjp(refined, values, _B)[1](_DB), jax.vjp(exact, values, _B)[1](_DB)):
        assert _relative(got, want) <= _EXACT
    hessian = jax.hessian(lambda rhs: jnp.sum(refined(values, rhs) ** 2))(_B)
    assert bool(jnp.all(jnp.isfinite(hessian)))


class _Inexact(Dense):
    backend_name = "inexact_dense"
    is_exact = False
    default_tolerance = Tolerance(**{"rtol": 1e-10, "atol": 1e-12}, max_iter=50)


def _route(topology, mode, backend=None):
    return linear_route(
        backend,
        LinearPlan(topology=topology, precision=ds.Precision(solve_mode=mode)),
        dtype=jnp.float64,
    )


def test_routes_fall_back_to_a_full_precision_factor_and_refuse_an_inexact_backend():
    """With every entry beyond float32's range the single-precision factorization fails: the
    reduced route's solve and linear solve fall back to double precision, so their values,
    tangents and cotangents equal the full-precision route's. Refinement and the route factor
    refuse an inexact backend's factor, naming it, while a route over it records that it is
    inexact."""
    topology, values, _ = _system(dirichlet=True)
    values = values * 1e39
    reduced, full = _route(topology, "fp32_ir"), _route(topology, "fp64")
    dv = 0.1 * values * jnp.asarray(np.random.default_rng(11).uniform(-1, 1, values.shape))
    for method in ("solve", "linear_solve"):

        def solved(route, method=method):
            return lambda v, rhs: getattr(RouteFactor(route, v), method)(rhs)

        got = jax.jvp(solved(reduced), (values, _B), (dv, _DB))
        want = jax.jvp(solved(full), (values, _B), (dv, _DB))
        pulled = zip(
            jax.vjp(solved(reduced), values, _B)[1](_DB), jax.vjp(solved(full), values, _B)[1](_DB)
        )
        assert max(_relative(g, w) for g, w in (*zip(got, want), *pulled)) <= _EXACT
    topology, values, _ = _system()
    analysis = analyze(topology, backend=_Inexact())
    assert factor(analysis, values).exact is False
    with pytest.raises(FactorContractError, match="inexact backend _Inexact"):
        refine(factor(analysis, values), _B, values=values)
    route = linear_route(None, analysis, dtype=jnp.float64)
    assert route.exact is False
    with pytest.raises(FactorContractError, match="_Inexact is inexact.*exact backend"):
        RouteFactor(route, values)


# The options the stock correctors plan an ``fp32_ir`` solve with: the refinement target and
# budget the route reads, and cuDSS's native refinement pinned off.
_FP32_IR_PLAN = (("ir_max_steps", 5), ("ir_nsteps", 0), ("ir_target_rel_residual", 0.05))


@pytest.mark.parametrize(
    "name,settings,mode,options",
    [
        ("Dense", {}, "fp64", ()),
        ("Dense", {}, "fp32_ir", ()),
        pytest.param("KLU", {}, "fp64", (), marks=pytest.mark.requires_klu),
        pytest.param("SuperLU", {}, "fp64", (), marks=pytest.mark.requires_superlu),
        pytest.param("CuDSS", {}, "fp64", (), marks=pytest.mark.requires_cudss),
        pytest.param("CuDSS", {}, "fp32_ir", (("ir_nsteps", 0),), marks=pytest.mark.requires_cudss),
        pytest.param(
            "CuDSS",
            {"ir_nsteps": 2},
            "fp32_ir",
            (("ir_nsteps", 0),),
            marks=pytest.mark.requires_cudss,
        ),
    ],
)
def test_a_route_analyzes_a_backend_without_settings_with_its_plans_options_alone(
    name, settings, mode, options
):
    """A backend built without settings, the one its name selects, adds no options: a route
    analyzes it with exactly its plan's options, under the memoisation key of a direct analyze
    of that backend, and an ``fp32_ir`` plan still sets the refinement target 0.05 and budget 5
    and passes ``ir_nsteps=0`` to cuDSS, over a setting of the same name."""
    topology = _system()[0]
    plan = LinearPlan(
        topology=topology,
        options=_FP32_IR_PLAN if mode == "fp32_ir" else (),
        precision=ds.Precision(solve_mode=mode),
    )
    route = linear_route(getattr(ds.linear, name)(**settings), plan, dtype=jnp.float64)
    backend = getattr(ds.linear, name)(**settings)
    assert route.full.options == options
    assert route.full is analyze(topology, backend=backend, options=dict(options) or None)
    if mode == "fp32_ir":
        assert (route.residual_target, route.max_steps, route.reduced.options) == (0.05, 5, options)


def _bordered():
    def rhs(t, y, theta, control):
        x0, x1, g, w = y
        return jnp.stack(
            [
                -x0,
                theta[0] * x0 + theta[1] * g + 0.3 * w + 0.25 - x1,
                -g,
                theta[2] + t - w - x1 - 0.5 * g,
            ]
        )

    return ds.MassProblem(
        rhs=rhs,
        mass=ds.MassOperator.sparse(
            [0, 2],
            [0, 2],
            lambda t, y, theta, control: jnp.ones((2,), y.dtype),
            n=4,
            dependencies=ds.Dependencies(time=False, state=False, parameters=False, control=False),
        ),
        structure=ds.DAEStructure(
            differential_vars=np.array([True, False, True, False]),
            partitions=(("state", 0, 2), ("global", 2, 4)),
        ),
    )


_PROBLEM, _STATE = _bordered(), jnp.asarray([1.3, 0.0, 0.8, 0.0])
_THETA = jnp.asarray([0.7, -0.4, 1.1])
_FORWARD = ds.AlgebraicProjectionOptions(
    max_steps=10, linear_solver="dense", ad_mode="forward", **{"rtol": 1e-12, "atol": 1e-14}
)


def _loss(theta, options=_FORWARD):
    state = ds.project_algebraic(
        _PROBLEM, t=0.4, state=_STATE, seed=jnp.zeros(4), theta=theta, options=options
    ).state
    return jnp.sum(jnp.asarray([0.2, 1.0, -0.4, 0.7]) * state) + state[1] * state[3]


def _closed_loss(theta):
    a, b, p = theta
    x1 = (a * _STATE[0] + (b - 0.15) * _STATE[2] + 0.3 * (p + 0.4) + 0.25) / 1.3
    w = p + 0.4 - x1 - 0.5 * _STATE[2]
    return 0.2 * _STATE[0] + x1 - 0.4 * _STATE[2] + 0.7 * w + x1 * w


def test_root_second_derivatives_match_the_closed_form_and_the_entries_agree():
    """The Hessian of a nonlinear loss of the projected root in the parameters of its right-hand
    side equals the closed form (forward entries), and the reverse entries give the forward
    entries' gradient. (A parameter that scales a state coupling of the algebraic rows is left
    out: its second derivative is not yet exact.)"""
    expected = jax.hessian(_closed_loss)(_THETA)
    assert _relative(jax.hessian(_loss)(_THETA), expected) <= _ROOT
    reverse = dataclasses.replace(_FORWARD, ad_mode="reverse")
    assert (
        _relative(jax.grad(lambda th: _loss(th, reverse))(_THETA), jax.grad(_loss)(_THETA)) <= _ROOT
    )


def test_a_jitted_root_gradient_traces_once_and_lowers_to_one_text():
    """A jitted gradient through the projection traces once across parameter values, and two
    lowerings give one program text: no per-call identity reaches the program."""
    traces = []

    def gradient(theta):
        traces.append(None)
        return jax.grad(_loss)(theta)

    compiled = jax.jit(gradient)
    compiled(_THETA).block_until_ready()
    compiled(1.01 * _THETA).block_until_ready()
    assert len(traces) == 1
    assert (
        jax.jit(gradient).lower(_THETA).as_text() == jax.jit(gradient).lower(1.1 * _THETA).as_text()
    )
