"""Approximators: bases from domains, sparse Legendre, networks, operators, learnable functions."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import diff_soup as ds
from diff_soup import inference as inf
from diff_soup.domains import cartesian_grid_1d, rectangle_mesh
from diff_soup.inference import approx
from tests.support.inference import small_fit, solver


@pytest.mark.parametrize("kind", ["grid", "mesh"])
def test_a_kl_basis_from_a_domain_reads_its_bounds_metric_and_nodes(kind):
    """KLBasis.from_domain is the nodes' basis, mass-orthonormal, sized and fingerprinted."""
    if kind == "grid":
        domain, bounds = cartesian_grid_1d(6, 0.0, 2.0, method="fvm"), ((0.0, 2.0),)
    else:
        domain, bounds = rectangle_mesh(1, 1, 2.0, 3.0, ele_type="QUAD4"), ((0.0, 2.0), (0.0, 3.0))
    assert domain.bounds == bounds
    basis = approx.KLBasis.from_domain(domain, length_scale=1.2, n_modes=3)
    raw = approx.KLBasis.from_mesh(
        np.asarray(domain.node_coordinates()), length_scale=1.2, n_modes=3
    )
    assert basis.num_points == domain.num_nodes and basis.domain_fingerprint
    longer = cartesian_grid_1d(6, 0.0, 3.0, method="fvm")
    other = approx.KLBasis.from_domain(longer, length_scale=1.2, n_modes=3)
    assert other.domain_fingerprint != basis.domain_fingerprint
    # measured 1.8e-15 and 6.7e-16, CPU VM 2026-10-22, pytest tests/public_api/test_approximators.py
    np.testing.assert_allclose(basis.basis @ basis.basis.T, raw.basis @ raw.basis.T, atol=1e-12)
    gram = basis.modes.T @ (basis.mass_diagonal[:, None] * basis.modes)
    np.testing.assert_allclose(gram, np.eye(3), atol=1e-12)
    np.testing.assert_array_equal(basis(domain.node_coordinates()), basis())
    if kind == "mesh":
        cosine = approx.GRFBasis.from_domain(domain, M=2)
        points = np.asarray(domain.node_coordinates())
        reference = approx.GRFBasis.from_mesh(points, 2.0, 3.0, M=2)
        # measured 0.0 difference, CPU VM 2026-10-22
        np.testing.assert_allclose(cosine.basis, reference.basis, atol=1e-12)
    with pytest.raises(ValueError, match=f"{domain.num_nodes} points"):
        basis(jnp.ones((domain.num_nodes - 1, len(bounds))))


def test_sparse_legendre_counts_orders_weights_and_reproduces_a_quadratic():
    """Total-degree index sets have C(d+n, n) graded entries and reproduce polynomials exactly."""
    for dimension, degree in ((1, 4), (2, 3), (3, 2)):
        basis = approx.SparseLegendre(intervals=((-1.0, 1.0),) * dimension, degree=degree)
        assert len(basis.indices) == math.comb(degree + dimension, dimension)
        assert basis.indices == tuple(sorted(basis.indices, key=lambda k: (sum(k), k)))
        np.testing.assert_array_equal(
            approx.sparse_legendre_weights(basis),
            [1 + sum(k) * (sum(k) + 1) for k in basis.indices],
        )
    coeffs = jnp.asarray([14 / 3, 3.0, 2.0, 4.0, 4.0, 10 / 3])
    quadratic = approx.SparseLegendre(intervals=((-1.0, 1.0),) * 2, degree=2, coeffs=coeffs)
    x, y = jnp.linspace(-1.0, 1.0, 5)[:, None], jnp.linspace(-1.0, 1.0, 7)[None, :]
    # measured 1.8e-15, CPU VM 2026-10-22, pytest tests/public_api/test_approximators.py
    np.testing.assert_allclose(
        quadratic(x, y), 1 + 2 * x + 3 * y + 4 * x * y + 5 * x * x + 6 * y * y, atol=1e-12
    )


def test_sparse_legendre_coefficients_of_a_two_field_law_are_recovered():
    """A degree-2 law D(c, T) fitted from two coupled fields recovers all six coefficients."""
    basis = approx.SparseLegendre(
        intervals=((0.0, 1.0), (250.0, 350.0)), degree=2, coeffs=jnp.asarray([0.25] + [0.0] * 5)
    )
    truth = jnp.asarray([0.3, 0.02, 0.03, 0.01, 0.01, 0.02])
    spec = ds.problem("approximators_two_field_law")
    spec.domain("rod", cartesian_grid_1d(4, 0.0, 1.0, method="fvm"))
    spec.field("c", mass="differential", scale=1.0)
    spec.field("T", mass="differential", scale=300.0)
    law = spec.function("D", basis, learnable=True)

    @spec.equation(writes="c", reads=("c", "T"))
    def reaction(q):
        return -q.fn.D(q.c, q.T) * q.c

    @spec.equation(writes="T", reads=("T",))
    def cooling(q):
        return -0.05 * (q.T - 260.0)

    model = spec.build()
    y0 = spec.assemble_state(
        {"c": jnp.asarray([0.25, 0.45, 0.65, 0.8]), "T": jnp.asarray([260.0, 310.0, 275.0, 340.0])}
    )
    times = jnp.linspace(0.0, 2.0, 8)
    theta = ds.default_theta(model, {law: truth})
    data = ds.pdesolve(model, solver(), (0.0, 2.0), y0, theta=theta, saveat=ds.SaveAt(ts=times)).ys
    experiment = inf.Experiment(
        model, solver(), (0.0, 2.0), y0, (inf.Measurement("state", times, data),)
    )
    fit = small_fit(inf.InferenceProblem(experiment, parameters=law), inf.LevenbergMarquardt(35))
    # measured 9.5e-5 worst relative coefficient error (the leading 0.30 to 1.9e-5),
    # CPU VM 2026-10-22, pytest tests/public_api/test_approximators.py
    np.testing.assert_allclose(fit[law].coeffs, truth, rtol=0.05)


@pytest.mark.parametrize("kind", ["FourierFeatureNet", "SIREN"])
def test_coordinate_nets_normalize_their_bounds_and_take_them_from_a_domain(kind):
    """A bounded coordinate net equals the plain net on [-1, 1]; from_domain reads the bounds."""
    cls = getattr(approx, kind)
    key = jax.random.PRNGKey(4)
    bounded = cls(2, width=3, depth=1, key=key, out_size=2, bounds=((2.0, 4.0), (1.0, 5.0)))
    plain = cls(2, width=3, depth=1, key=key, out_size=2)
    points = jnp.asarray([[2.0, 1.0], [3.0, 3.0], [4.0, 5.0]])
    # measured 0.0 difference, CPU VM 2026-10-22, pytest tests/public_api/test_approximators.py
    normalized = jnp.asarray([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(bounded(points), plain(normalized), atol=1e-12)
    assert bounded(points).shape == (3, 2) and bounded(points[0]).shape == (2,)
    assert all(path.startswith("mlp.layers.") for path in bounded.learnable_paths())
    rod = cartesian_grid_1d(6, 2.0, 4.0, method="fvm")
    assert cls.from_domain(rod, width=3, depth=1, key=key).bounds == ((2.0, 4.0),)


def test_structured_networks_keep_convexity_monotonicity_and_parity():
    """ICNN is convex, MonotoneMLP monotone per sign, and Transform 'odd' is odd."""
    convex = approx.ICNN(2, width=5, depth=3, key=jax.random.PRNGKey(1))
    pairs = jax.random.normal(jax.random.PRNGKey(2), (2, 200, 2))
    midpoint = convex((pairs[0] + pairs[1]) / 2)
    assert bool(jnp.all(midpoint <= (convex(pairs[0]) + convex(pairs[1])) / 2 + 1e-12))
    monotone = approx.MonotoneMLP(
        3, width=4, depth=2, monotone_in=(1, -1, 0), key=jax.random.PRNGKey(3)
    )
    points = jax.random.normal(jax.random.PRNGKey(4), (200, 3))
    for axis, sign in ((0, 1.0), (1, -1.0)):
        change = monotone(points.at[:, axis].add(1e-3)) - monotone(points)
        assert bool(jnp.all(sign * change >= -1e-12))
    odd = approx.Transform(eqx.nn.MLP("scalar", "scalar", 3, 1, key=jax.random.PRNGKey(5)), "odd")
    grid = jnp.linspace(-2.0, 2.0, 7)
    np.testing.assert_array_equal(jax.vmap(odd)(-grid), -jax.vmap(odd)(grid))
    assert len(odd.learnable_paths()) == 4
    with pytest.raises(ValueError, match="convex.*nondecreasing"):
        approx.ICNN(1, width=2, depth=1, key=jax.random.PRNGKey(0), activation="tanh")


def test_a_fourier_operator_trains_on_field_pairs_and_runs_inside_a_recipe():
    """FNO1d trained with optax lowers its loss; a recipe takes it on a uniform grid only."""
    grid = cartesian_grid_1d(6, 0.0, 1.0, method="fvm")
    x = grid.node_coordinates()
    coefficients = jax.random.uniform(jax.random.PRNGKey(5), (20, 2), minval=0.1, maxval=0.8)
    samples = coefficients[:, :1] + coefficients[:, 1:] * jnp.sin(2 * jnp.pi * x[:, 0])
    targets = 0.2 + 0.05 * samples
    network = approx.FNO1d(modes=3, width=4, depth=1, key=jax.random.PRNGKey(6))
    optimizer = optax.adam(0.01)
    state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(net, optimizer_state):
        def loss(candidate):
            return jnp.mean((jax.vmap(candidate)(samples) - targets) ** 2)

        value, gradient = eqx.filter_value_and_grad(loss)(net)
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, net)
        return eqx.apply_updates(net, updates), optimizer_state, value

    losses = []
    for _ in range(40):
        network, state, value = step(network, state)
        losses.append(float(value))
    # measured a 40-step drop to 0.001 of the first loss, CPU VM 2026-10-22,
    # pytest tests/public_api/test_approximators.py
    assert losses[-1] < 0.1 * losses[0]
    accepted = ds.problem("approximators_operator").domain("rod", grid)
    assert accepted.function("K", network, learnable=True) is not None
    uneven = eqx.tree_at(lambda g: g.nodes, grid, grid.nodes.at[3].add(0.02))
    refused = ds.problem("approximators_uneven").domain("uneven_rod", uneven)
    with pytest.raises(ValueError, match="uniform.*uneven_rod"):
        refused.function("K", network, learnable=True)


def test_a_learnable_signal_drives_a_dirichlet_boundary_through_theta():
    """A PiecewiseLinear boundary signal pins the boundary to its knots, and new knot values
    passed as theta drive the same compiled solve."""
    signal = approx.PiecewiseLinear([0.0, 0.5, 1.0], [0.4, 1.0, 0.2])
    spec = ds.problem("approximators_boundary_signal")
    spec.domain("line", cartesian_grid_1d(3, 0.0, 1.0, method="fvm"))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: 0.0 * x)
    inlet = spec.function("inlet", signal, learnable=True)

    @spec.equation(writes="u", reads=("u",))
    def stationary(q):
        return 0.0 * q.u + 0.0 * q.fn.inlet(q.t)

    spec.bc.dirichlet("u", node_indices=[0], value_fn=signal)
    model, y0 = spec.build(), spec.assemble_state({})
    times = jnp.asarray([0.25, 0.5, 1.0])

    def boundary(knots):
        theta = ds.default_theta(model, {inlet: knots})
        return ds.pdesolve(
            model, solver(), (0.0, 1.0), y0, theta=theta, saveat=ds.SaveAt(ts=times)
        ).ys[:, 0]

    # measured 2.4e-6 at the knot, CPU VM 2026-10-22, pytest tests/public_api/test_approximators.py
    np.testing.assert_allclose(boundary(jnp.asarray([0.4, 1.0, 0.2])), [0.7, 1.0, 0.2], atol=3e-5)
    np.testing.assert_allclose(boundary(jnp.asarray([0.0, 2.0, 1.0])), [1.0, 2.0, 1.0], atol=3e-5)


class _DeclaredMLP(eqx.Module):
    mlp: eqx.nn.MLP

    def __call__(self, x):
        return self.mlp(x)

    def learnable_paths(self):
        return ("mlp.layers.0.weight", "mlp.layers.0.bias")


def _reaction_problem(module, name):
    spec = ds.problem(name)
    spec.domain("line", cartesian_grid_1d(3, 0.0, 1.0, method="fvm"))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: 1.0 + x)
    handle = spec.function("reaction", module, learnable=True, pointwise=True)

    @spec.equation(writes="u", reads=("u",))
    def reaction(q):
        return q.fn.reaction(q.u)

    y0 = spec.assemble_state({})
    times = jnp.linspace(0.0, 0.8, 9)
    data = jax.vmap(lambda t: (y0 - 0.4) * jnp.exp(-0.5 * t) + 0.4)(times)
    experiment = inf.Experiment(
        spec.build(), solver(), (0.0, 0.8), y0, (inf.Measurement("u", times, data),)
    )
    return inf.InferenceProblem(experiment, parameters=handle)


def test_a_declared_function_module_fits_its_declared_parameters():
    """Bare and declared MLPs expose the same two parameters; the declared one fits u' = -0.5u+0.2."""
    bare = _reaction_problem(
        eqx.nn.MLP("scalar", "scalar", 1, 0, key=jax.random.PRNGKey(0)), "approximators_bare"
    )
    declared = _reaction_problem(
        _DeclaredMLP(eqx.nn.MLP("scalar", "scalar", 1, 0, key=jax.random.PRNGKey(0))),
        "approximators_declared",
    )
    assert bare.n_params == declared.n_params == 2
    fit = small_fit(declared, inf.LevenbergMarquardt(max_steps=80))
    # measured 1.2e-6, CPU VM 2026-10-22, pytest tests/public_api/test_approximators.py
    np.testing.assert_allclose(fit.params, [-0.5, 0.2], atol=2e-5)


def test_a_learned_initial_state_is_fitted_with_the_rate():
    """An Experiment with LearnedInitialState fits its start state and its rate together."""
    spec = ds.problem("approximators_learned_start")
    spec.domain("line", cartesian_grid_1d(2, 0.0, 1.0, method="fvm"))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: 0.0 * x + 1.0)
    rate = spec.param("rate", value=0.1, learnable=True)

    @spec.equation(writes="u", reads=("u",))
    def decay(q):
        return -q.p.rate * q.u

    times = jnp.linspace(0.0, 1.0, 6)
    target = 2.0 * jnp.exp(-0.3 * times[:, None]) * jnp.ones((1, 2))
    experiment = inf.Experiment(
        spec.build(),
        solver(),
        (0.0, 1.0),
        inf.LearnedInitialState(jnp.asarray([1.0, 1.0])),
        (inf.Measurement("u", times, target),),
        id="learned",
    )
    fit = small_fit(inf.InferenceProblem(experiment, parameters=rate), inf.LevenbergMarquardt(60))
    # measured 3.1e-6 on the rate and 1.5e-6 on the start, CPU VM 2026-10-22,
    # pytest tests/public_api/test_approximators.py
    np.testing.assert_allclose(fit.params, [0.3, 2.0, 2.0], atol=2e-4)
