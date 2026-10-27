"""Implicit layer: root derivatives by the implicit-function rule (the exact sensitivity matrix
at the converged root, never the Newton iterations), consistent initialization and projection
derivatives, and the content-keyed, bounded solver-view cache."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

import diff_soup as ds
from diff_soup.problem.implicit import (
    IterationMatrixCertificate,
    consistent_initial_state,
    get_or_build_solver_view,
    implicit_step_root,
)
from tests.support.problems import DecayODE

_OK, _T, _CJ = jnp.asarray(True), jnp.asarray(0.4), jnp.asarray(2.5)
_Q, _GUESS = jnp.asarray([0.1, -0.2]), jnp.zeros(2)
_THETA = jnp.asarray([1.3, 0.7, 0.2])
# measured at most 7.7e-13 relative (the projection solved to rtol 1e-11; 2e-16 for the step
# root, the initialization and the discarded overflowing attempt, 0.0 for the certified reuse),
# CPU VM 2026-10-22, pytest tests/contracts/test_implicit_layer.py
_ROOT_BOUND = 1e-11
# measured 1.1e-16 absolute between the reverse and forward initialization Jacobians, CPU VM
# 2026-10-22, pytest tests/contracts/test_implicit_layer.py::test_consistent_initialization_solves_the_joint_root_with_exact_derivatives
_FLOOR = 1e-14
_FAILED = jnp.asarray(False)
#: Unit mass on the first of two rows: one differential and one algebraic unknown.
_FIRST_ROW_MASS = ds.MassOperator.sparse(
    [0],
    [0],
    lambda t, y, theta, control: jnp.ones((1,), y.dtype),
    n=2,
    dependencies=ds.Dependencies(time=False, state=False, parameters=False, control=False),
)
# measured 9.7e-16 relative (forward over forward; 4.8e-16 for jax.hessian in both modes), the
# root solved to rtol 1e-13, workstation 2026-10-23,
# pytest tests/contracts/test_implicit_layer.py::test_the_projection_hessian_differentiates_the_root_by_its_rule_again
_SECOND_ORDER = 1e-12


def _step_problem(rhs=None, parametric_mass=True):
    mass = ds.MassOperator.diagonal(
        lambda t, y, theta, control: (
            (theta[0] if parametric_mass else 1.0) * jnp.asarray([1.0, 2.0])
        ),
        n=2,
        dependencies=ds.Dependencies(
            time=False, state=False, parameters=parametric_mass, control=False
        ),
    )
    return ds.MassProblem(
        rhs=rhs
        or (
            lambda t, y, theta, control: jnp.stack(
                [-theta[1] * y[0] * y[1] + jnp.sin(t), y[0] - y[1] ** 2 + theta[2]]
            )
        ),
        mass=mass,
        structure=ds.DAEStructure(differential_vars=np.array([True, True])),
    )


_PROBLEM = _step_problem()
_PLAN = get_or_build_solver_view(_PROBLEM).plan
# The residual and its parameter partials overflow once y[0] passes about 709, as the iterate
# of a diverged corrector can.
_OVERFLOWING = _step_problem(
    lambda t, y, theta, control: jnp.stack(
        [theta[2] - theta[0] * jnp.exp(y[0]), theta[1] * y[0] - y[1]]
    ),
    parametric_mass=False,
)


def _newton(iters=60, chord=False, shift=0.0, problem=_PROBLEM):
    """A primal root solve of ``F(t, y, cj y + q) = 0`` by full, chord or shifted Newton."""

    def primal(t, cj, q, theta, control, guess):
        def g(y):
            return problem.residual(t, y, cj * y + q, theta, control)

        frozen = jax.jacfwd(g)(guess)

        def body(_, y):
            j = frozen if chord else jax.jacfwd(g)(y)
            return y - jnp.linalg.solve(j + shift * jnp.eye(2), g(y))

        return jax.lax.fori_loop(0, iters, body, guess)

    return primal


_ENTRIES = {}


def _root(primal, theta=_THETA, t=_T, cj=_CJ, q=_Q, guess=_GUESS, ok=_OK, problem=_PROBLEM):
    """The root of a bare primal solve through the one derivative entry; ``ok`` reaches the
    entry as gradient-dead primal state, never through the primal's closure."""
    if primal not in _ENTRIES:
        _ENTRIES[primal] = lambda t, cj, q, th, control, guess, valid: (
            primal(t, cj, q, th, control, guess),
            valid,
            {},
        )
    plan = get_or_build_solver_view(problem).plan
    head = (plan, None, None, _ENTRIES[primal], None, t, cj, q, theta, None, guess, ok)
    return implicit_step_root(*head)[0]


_FULL = _newton()


def test_a_step_root_derivative_ignores_the_iteration_gates_failures_and_reuses_certified_matrices():
    """Full, chord and shifted Newton give one gradient, forward mode and its transpose agree
    and the time, coefficient and guess are dead; an unaccepted root has NaN tangents and gives
    its inputs exact zero cotangents; a held matrix is reused only when certified at the root,
    for this coefficient and exact."""
    grads = [
        jax.grad(lambda th, p=p: jnp.sum(_root(p, th) ** 2))(_THETA)
        for p in (_FULL, _newton(250, chord=True), _newton(250, shift=0.15))
    ]
    np.testing.assert_allclose(grads[1:], [grads[0]] * 2, rtol=_ROOT_BOUND)
    forward = jax.jacfwd(lambda th: _root(_FULL, th))(_THETA)
    reverse = jax.jacrev(lambda th: _root(_FULL, th))(_THETA)
    np.testing.assert_allclose(reverse, forward, rtol=_ROOT_BOUND)
    dead = jax.grad(
        lambda t, cj, g: jnp.sum(_root(_FULL, t=t, cj=cj, guess=g) ** 2), argnums=(0, 1, 2)
    )(_T, _CJ, _GUESS)
    assert float(dead[0]) == 0.0 and float(dead[1]) == 0.0
    np.testing.assert_array_equal(dead[2], 0.0)
    _, dy = jax.jvp(lambda q: _root(_FULL, q=q, ok=_FAILED), (_Q,), (jnp.ones(2),))
    assert bool(jnp.all(jnp.isnan(dy)))
    cotangent = jax.grad(lambda th: jnp.sum(_root(_FULL, th, ok=_FAILED) ** 2))(_THETA)
    np.testing.assert_array_equal(cotangent, 0.0)
    y_star = _FULL(_T, _CJ, _Q, _THETA, None, _GUESS)
    doubled = 2.0 * _PLAN.combined_dense(_T, y_star, _CJ * y_star + _Q, _THETA, None, _CJ)

    def primal(t, cj, q, theta, control, guess, args):
        return _FULL(t, cj, q, theta, control, guess), _OK, {"values": doubled}

    def gradient(cj=_CJ, at_root=True, exact=True, certify=True):
        def extractor(aux, *args):
            return IterationMatrixCertificate(
                values=aux["values"], cj=cj, at_root=jnp.asarray(at_root), exact=exact
            )

        def loss(theta):
            head = (_PLAN, None, None, primal, None, _T, _CJ, _Q, theta, None, _GUESS, None)
            y, _, _ = implicit_step_root(*head, extractor if certify else None)
            return jnp.sum(y**2)

        return jax.grad(loss)(_THETA)

    plain = gradient(certify=False)
    np.testing.assert_allclose(gradient(), 0.5 * plain, rtol=_ROOT_BOUND)  # the doubled matrix ran
    for rejected in (gradient(cj=1.5 * _CJ), gradient(at_root=False), gradient(exact=False)):
        np.testing.assert_allclose(rejected, plain, rtol=_ROOT_BOUND)


def test_a_discarded_attempt_at_an_overflowing_iterate_leaves_the_gradient_finite():
    """A retry ladder keeps the converged retry of a diverged attempt whose iterate overflows
    the residual: the discarded attempt contributes exact zeros, not zero times an infinite
    partial, so both modes give the retry's own derivative."""
    converged = _newton(problem=_OVERFLOWING)
    diverged = jnp.asarray([800.0, -3.0])

    def ladder(theta):
        first = _root(lambda *a: diverged, theta, ok=_FAILED, problem=_OVERFLOWING)
        retry = _root(converged, theta, problem=_OVERFLOWING)
        finite = jnp.isfinite(_OVERFLOWING.residual(_T, first, _CJ * first + _Q, theta, None))
        return jnp.sum(jnp.where(jnp.all(finite), first, retry) ** 2)

    expected = jax.jacfwd(lambda th: jnp.sum(_root(converged, th, problem=_OVERFLOWING) ** 2))(
        _THETA
    )
    np.testing.assert_allclose(
        [jax.jacfwd(ladder)(_THETA), jax.grad(ladder)(_THETA)], [expected] * 2, rtol=_ROOT_BOUND
    )


def _coupled():
    """Two concentrations and an algebraic rate read from their slopes (closed-form root)."""
    mass = ds.MassOperator.block_sparse(
        blocks=(((0, 1), (0, 1), jnp.eye(2)),),
        couplings=((2, (0, 1), lambda t, y, theta, control: theta["w"]),),
        n=3,
        dependencies=ds.Dependencies(time=False, state=False, parameters=True, control=False),
    )
    return ds.MassProblem(
        rhs=lambda t, y, theta, control: jnp.stack(
            [y[2] ** 2 - theta["k"] * y[0], y[2] - theta["k"] * y[1] ** 2, y[2]]
        ),
        mass=mass,
        structure=ds.DAEStructure(differential_vars=np.array([True, True, False])),
    )


def _closed_start(theta, y0):
    k, w = theta["k"], theta["w"]
    b, c = w[1] - 1.0, -k * (w[0] * y0[0] + w[1] * y0[1] ** 2)
    rate = (-b - jnp.sqrt(b**2 - 4.0 * w[0] * c)) / (2.0 * w[0])
    return jnp.stack([y0[0], y0[1], rate, rate**2 - k * y0[0], rate - k * y0[1] ** 2, 0.0])


def test_consistent_initialization_solves_the_joint_root_with_exact_derivatives():
    """The joint initialization holds the differential values, solves the rate and slopes to
    the closed form, differentiates exactly in both modes (the algebraic guess is dead); a root that does not
    exist has NaN tangents and gives its inputs exact zero cotangents."""
    problem, y0 = _coupled(), jnp.asarray([1.2, 0.9, 0.0])
    theta = {"k": jnp.asarray(0.8), "w": jnp.asarray([0.25, 0.5])}

    def start(y, th):
        result = consistent_initial_state(problem, jnp.asarray(0.0), y, th)
        return jnp.concatenate([result.y, result.ydot])

    np.testing.assert_allclose(start(y0, theta), _closed_start(theta, y0), rtol=_ROOT_BOUND)
    forward = jax.jacfwd(lambda y: start(y, theta))(y0)
    np.testing.assert_allclose(
        forward[:, :2], jax.jacfwd(lambda y: _closed_start(theta, y))(y0)[:, :2], rtol=_ROOT_BOUND
    )
    np.testing.assert_array_equal(forward[:, 2], 0.0)
    np.testing.assert_allclose(
        jax.jacrev(lambda y: start(y, theta))(y0), forward, rtol=_ROOT_BOUND, atol=_FLOOR
    )
    by_theta = jax.jacrev(lambda th: start(y0, th))(theta)
    expected = jax.jacfwd(lambda th: _closed_start(th, y0))(theta)
    np.testing.assert_allclose(by_theta["w"], expected["w"], rtol=_ROOT_BOUND)
    bad = {"k": jnp.asarray(-2.0), "w": theta["w"]}
    failed = jnp.asarray([1.0, 1.0, 0.0])

    def unsolvable(th):
        return consistent_initial_state(problem, 0.0, failed, th, max_iters=25).y

    assert bool(jnp.all(jnp.isnan(jax.jacfwd(unsolvable)(bad)["k"])))
    cotangent = jax.grad(lambda th: jnp.sum(unsolvable(th)))(bad)
    np.testing.assert_array_equal(jnp.concatenate([cotangent["k"][None], cotangent["w"]]), 0.0)


def test_an_initialization_failing_at_an_infinite_partial_gives_exact_zero_cotangents():
    """An initialization whose root does not exist at a point where the residual's partial is
    infinite (``sqrt(y)`` at ``y = 0``) has NaN tangents and gives its pinned state and its
    parameters exact zero cotangents, on the joint and the reduced route."""
    problem = ds.MassProblem(
        rhs=lambda t, y, theta, control: jnp.stack(
            [-theta[0] * y[0], jnp.exp(y[1]) - 1.0 - theta[0] * jnp.sqrt(y[0])]
        ),
        mass=_FIRST_ROW_MASS,
        structure=ds.DAEStructure(differential_vars=np.array([True, False])),
    )
    y0, k = jnp.asarray([0.0, 0.5]), jnp.asarray([0.7])
    for route in ("joint", "reduced"):

        def start(y, th, route=route):
            return jnp.sum(consistent_initial_state(problem, 0.0, y, th, route=route).y)

        assert bool(jnp.all(jnp.isnan(jax.jacfwd(start, argnums=1)(y0, k))))
        cotangents = jax.grad(start, argnums=(0, 1))(y0, k)
        np.testing.assert_array_equal(jnp.concatenate(cotangents), 0.0)


def _bordered():
    def rhs(t, y, theta, control):
        x0, x1, g, w = y
        a, b, c, d, e, p = (theta[k] for k in "abcdep")
        return jnp.stack([-x0, a * x0 + b * g + c * w + d - x1, -g, p + t - w - x1 - e * g])

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


def test_the_projection_differentiates_by_its_implicit_rules_in_both_modes():
    """Forward and reverse derivatives of the projection through time, the pinned state and
    the parameters equal the closed-form implicit solution, and the seed is dead in both."""
    problem = _bordered()
    theta = {k: jnp.asarray(v) for k, v in zip("abcdep", (0.7, -0.4, 0.3, 0.25, 0.5, 1.1))}
    t0, state, seed = jnp.asarray(0.4), jnp.asarray([1.3, 0.0, 0.8, 0.0]), jnp.zeros(4)
    forward = ds.AlgebraicProjectionOptions(
        max_steps=10, linear_solver="dense", ad_mode="forward", **{"rtol": 1e-11, "atol": 1e-13}
    )
    reverse = dataclasses.replace(forward, ad_mode="reverse")

    def projected(t, s, th, options, start=seed):
        return ds.project_algebraic(
            problem, t=t, state=s, seed=start, theta=th, options=options
        ).state

    def closed(t, s, th):
        a, b, c, d, e, p = (th[k] for k in "abcdep")
        x1 = (a * s[0] + (b - c * e) * s[2] + c * (p + t) + d) / (1.0 + c)
        return jnp.stack([s[0], x1, s[2], p + t - x1 - e * s[2]])

    tangents = (
        jnp.asarray(1.0),
        jnp.asarray([0.3, 0.0, -0.7, 0.0]),
        jax.tree_util.tree_map(jnp.ones_like, theta),
    )
    _, tangent = jax.jvp(lambda *a: projected(*a, forward), (t0, state, theta), tangents)
    np.testing.assert_allclose(
        tangent, jax.jvp(closed, (t0, state, theta), tangents)[1], rtol=_ROOT_BOUND
    )
    weights = jnp.asarray([0.2, 1.0, -0.4, 0.7])
    grads = jax.grad(lambda *a: jnp.sum(weights * projected(*a, reverse)), argnums=(0, 1))(
        t0, state, theta
    )
    expected = jax.grad(lambda *a: jnp.sum(weights * closed(*a)), argnums=(0, 1))(t0, state, theta)
    np.testing.assert_allclose(
        jnp.concatenate([grads[0][None], grads[1]]),
        jnp.concatenate([expected[0][None], expected[1]]),
        rtol=_ROOT_BOUND,
        atol=_FLOOR,
    )
    np.testing.assert_array_equal(
        jax.jvp(lambda s: projected(t0, state, theta, forward, s), (seed,), (jnp.ones(4),))[1], 0.0
    )
    np.testing.assert_array_equal(
        jax.grad(lambda s: jnp.sum(projected(t0, state, theta, reverse, s)))(seed), 0.0
    )


def test_the_projection_hessian_differentiates_the_root_by_its_rule_again():
    """The Hessian of a projected algebraic root equals the closed form of ``z^3 + z = k y``
    in both modes, the curvature term ``-g_zz (dz/dp)^2 / g_z`` included: the rules take their
    primal through the projection entry and linearize at the root without cutting it."""
    problem = ds.MassProblem(
        rhs=lambda t, y, theta, control: jnp.stack(
            [-theta[0] * y[0], y[1] ** 3 + y[1] - theta[0] * y[0]]
        ),
        mass=_FIRST_ROW_MASS,
        structure=ds.DAEStructure(differential_vars=np.array([True, False])),
    )
    point = jnp.asarray([1.3, 0.8])
    for mode in ("forward", "reverse"):
        options = ds.AlgebraicProjectionOptions(ad_mode=mode, **{"rtol": 1e-13, "atol": 1e-14})

        def root(p, options=options):
            state = jnp.stack([p[1], 0.0])
            return ds.project_algebraic(
                problem, t=0.0, state=state, seed=jnp.zeros(2), theta=p[:1], options=options
            ).state[1]

        z = root(point)
        slope, gradient = 1.0 / (3.0 * z**2 + 1.0), point[::-1]
        expected = -6.0 * z * slope**3 * jnp.outer(gradient, gradient) + slope * jnp.eye(2)[::-1]
        np.testing.assert_allclose(jax.hessian(root)(point), expected, rtol=_SECOND_ORDER)
        if mode == "forward":
            twice = jax.jacfwd(jax.jacfwd(root))(point)
            np.testing.assert_allclose(twice, expected, rtol=_SECOND_ORDER)


def _scaled_decay(scale):
    return ds.MassProblem(
        rhs=lambda t, y, theta, control: -scale * theta * y,
        mass=ds.MassOperator.identity(2),
        structure=ds.DAEStructure(differential_vars=np.ones((2,), dtype=bool)),
    )


def test_the_solver_view_cache_is_keyed_on_content_and_bounded():
    """A rebuilt equal problem is served the same view and one with another captured constant
    is not; viewing many distinct problems evicts the oldest view, whose rebuild compares equal
    to it, while the latest view stays cached."""
    rebuilt = get_or_build_solver_view(DecayODE().problem) is get_or_build_solver_view(
        DecayODE().problem
    )
    changed = get_or_build_solver_view(_scaled_decay(2.0)) is get_or_build_solver_view(
        _scaled_decay(3.0)
    )
    problems = [_scaled_decay(float(k)) for k in range(300)]
    first = get_or_build_solver_view(problems[0])
    views = [get_or_build_solver_view(problem) for problem in problems]
    again = get_or_build_solver_view(problems[0])
    assert (rebuilt, changed, again is first, again == first) == (True, False, False, True)
    assert get_or_build_solver_view(problems[-1]) is views[-1]
