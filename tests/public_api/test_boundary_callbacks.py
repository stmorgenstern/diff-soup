"""Boundary callbacks: flux and Dirichlet functions, facet totals and tractions, nodal access,
per-member flux parameters and declaration conflicts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup import inference as inf
from diff_soup.inference.approx import PiecewiseLinear

#: Solver settings (requested tolerances, not asserted bounds).
_TIGHT = {"rtol": 1e-8, "atol": 1e-10}
# tests/public_api/test_boundary_callbacks.py::test_a_spherical_flux_callback_equals_the_surface_source_and_its_rate_derivative
# tests/public_api/test_boundary_callbacks.py::test_callback_declarations_refuse_two_value_forms_and_unknown_names
# tests/public_api/test_boundary_callbacks.py::test_a_facet_flux_closes_its_total_and_a_traction_reads_the_facet_normal
# tests/public_api/test_boundary_callbacks.py::test_a_member_flux_reads_member_and_problem_parameters
# measured 0.0 on the residual, derivative, Robin, facet-total, traction and member-flux
# comparisons it bounds, CPU VM 2026-10-22, pytest on the four node ids above; measured 0.0
# on test_a_flux_on_a_clockwise_mesh_equals_the_flux_on_its_mirror, workstation CPU 2026-10-28
_EXACT = 1e-12
# measured 1.3e-8 relative between the fitted and the generating knot values, CPU VM 2026-10-22,
# pytest tests/public_api/test_boundary_callbacks.py::test_a_dirichlet_function_callback_fits_a_learnable_signal
_FIT_BOUND = 1e-6


def _particle(source=False, residual_scale=None, kind="flux"):
    """A four-cell sphere with surface uptake ``k (1 - c)`` as a flux callback (``kind="flux"``)
    or a source, or with the Robin condition ``c / 2 + 2 n.grad(c) = 1`` on its surface
    (``kind="robin"``)."""
    grid = ds.domains.spherical_grid_1d(4, 1.0, method="fvm")
    spec = ds.problem("uptake_source" if source else f"uptake_{kind}").domain("particle", grid)
    spec.field(
        "c",
        mass="differential",
        scale=1.0,
        residual_scale=residual_scale,
        init=lambda x: 0.2 + 0 * x,
    )
    k = spec.param("k", value=0.1, learnable=True)
    surface = jnp.zeros(4).at[-1].set(grid.face_areas[-1] / grid.cell_volumes[-1])

    @spec.equation(writes="c", reads=("c",))
    def diffusion(q):
        rhs = q.ops.laplacian(q.c)
        return rhs + q.p.k * (1 - q.c[-1]) * surface if source else rhs

    if kind == "robin":
        spec.bc.robin("c", node_indices=[3], alpha=0.5, beta=2.0, value=1.0)
    elif not source:
        spec.bc.flux("c", node_indices=[3], fn=lambda q, t: q.p.k * (1 - q.c))
    return spec, k


def _plate(cross_facet=False, residual_scale=None):
    """A 2 x 2 triangle plate with a facet flux ``k V (1 - c)`` on its right edge, named in a
    ``flux_total`` that closes ``V``, and a traction ``k n`` on its top edge."""
    mesh = ds.domains.rectangle_mesh(2, 2, 1.0, 1.0, ele_type="TRI3", with_boundary_tags=True)
    spec = ds.problem(f"facets_{cross_facet}").domain("plate", mesh)
    spec.field("c", mass="differential", scale=1.0, quad=False, residual_scale=residual_scale)
    spec.field(
        "u", mass="differential", scale=1.0, components=2, quad=False, residual_scale=residual_scale
    )
    spec.param("k", value=2.0, learnable=True)
    spec.global_var("V", initial=0.5)
    spec.equation(writes="c", reads=("c",))(lambda q: 0.0 * q.c)
    spec.equation(writes="u", reads=("u",))(lambda q: 0.0 * q.u)
    spec.bc.flux(
        "c",
        tag="right",
        name="current",
        cross_facet=cross_facet,
        fn=lambda q, t: q.p.k * q.V * (1 - q.c),
    )
    spec.constraint.flux_total(
        name="current_total", closes="V", from_bc="current", target_fn=lambda t: 0.8
    )
    spec.bc.traction("u", tag="top", fn=lambda q, t: q.p.k * q.geom.normal)
    return spec.build()


def _fit(problem, y0, times, data, handle, field):
    experiment = inf.Experiment(
        problem,
        ds.JNDF.stock(linear="dense", **_TIGHT),
        (0.0, float(times[-1])),
        y0,
        (inf.Measurement(field, times, data),),
    )
    return inf.optsolve(
        inf.InferenceProblem(experiment, parameters=handle),
        inf.LevenbergMarquardt(max_steps=30),
        diagnostics=False,
    )


def test_a_spherical_flux_callback_equals_the_surface_source_and_its_rate_derivative():
    """The flux callback ``k (1 - c)`` gives the residual of the same uptake written as a
    surface source, and the same derivative in ``k``."""
    (spec, k), (source_spec, source_k) = _particle(), _particle(source=True)
    state = jnp.linspace(0.2, 0.5, 4)

    def residual(problem, handle, rate):
        theta = ds.default_theta(problem, {handle: rate})
        return problem.solver_problem().residual(jnp.asarray(0.0), state, jnp.zeros(4), theta, None)

    flux, source = spec.build(), source_spec.build()
    for transform in (lambda f: f, jax.jacfwd):
        np.testing.assert_allclose(
            transform(lambda r: residual(flux, k, r))(0.3),
            transform(lambda r: residual(source, source_k, r))(0.3),
            atol=_EXACT,
        )


@pytest.mark.parametrize("kind", ["flux", "robin", "traction"])
def test_a_residual_scale_divides_a_boundary_condition_with_the_rest_of_its_row(kind):
    """With ``residual_scale=4`` every residual row of the field is the unscaled row divided
    by 4 (a power of two, so the division is exact), the rows a boundary flux, a Robin
    condition or a traction adds to included."""
    if kind == "traction":
        plain, scaled = _plate(), _plate(residual_scale=4.0)
        state, lam = plain.assemble_state({"c": 0.2, "u": 0.0}), jnp.array([0.5])
    else:
        plain, scaled = (_particle(residual_scale=s, kind=kind)[0].build() for s in (None, 4.0))
        state, lam = jnp.linspace(0.2, 0.5, 4), jnp.zeros(0)

    def residual(problem):
        return problem.physics_residual(state, lam, jnp.asarray(0.0))[0]

    np.testing.assert_array_equal(residual(scaled), residual(plain) / 4.0)


def test_a_dirichlet_function_callback_fits_a_learnable_signal():
    """A learnable piecewise-linear signal read by a Dirichlet ``fn`` callback fits its knot
    values to the boundary data."""
    spec = ds.problem("signal_boundary").domain(
        "rod", ds.cartesian_grid_1d(3, 0.0, 1.0, method="fvm")
    )
    spec.field("u", mass="differential", scale=1.0)
    inlet = spec.function(
        "inlet", PiecewiseLinear([0.0, 0.5, 1.0], [0.0, 0.0, 0.0]), learnable=True
    )
    spec.equation(writes="u", reads=("u",))(lambda q: 0 * q.u)
    spec.bc.dirichlet("u", node_indices=[0], fn=lambda q, t: q.fn.inlet(t))
    knots, times, target = (
        jnp.array([0.0, 0.5, 1.0]),
        jnp.linspace(0, 1, 9),
        jnp.array([0.4, 1.0, 0.2]),
    )
    data = jnp.zeros((9, 3)).at[:, 0].set(jnp.interp(times, knots, target))
    fitted = _fit(spec.build(), jnp.zeros(3), times, data, inlet, "u")[inlet]
    np.testing.assert_allclose(jax.vmap(fitted)(knots), target, rtol=_FIT_BOUND)


def test_callback_declarations_refuse_two_value_forms_and_unknown_names():
    """Each condition refuses two of ``value``, ``value_fn`` and ``fn`` together; a nodal and
    geometric Robin callback reads ``q.nodal`` and ``q.geom``; reading an undeclared name is
    a lowering error naming it."""
    spec = ds.problem("boundary_arguments")
    for kind in ("dirichlet", "flux", "robin", "traction"):
        extra = {"alpha": 1.0, "beta": 1.0} if kind == "robin" else {}
        for pair in (
            {"value": 1.0, "fn": lambda q, t: 1.0},
            {"value_fn": lambda t: 1.0, "fn": lambda q, t: 1.0},
            {"value": 1.0, "value_fn": lambda t: 1.0},
        ):
            with pytest.raises(ValueError, match="mutually exclusive"):
                getattr(spec.bc, kind)("u", node_indices=[0], **extra, **pair)
    particle, _ = _particle()
    particle.bc.robin(
        "c",
        node_indices=[0],
        alpha=0.0,
        beta=1.0,
        fn=lambda q, t: q.nodal.c[0] - q.c[0] + q.geom.normal[0, 0] + 1.0,
    )
    residual = particle.build().physics_residual(jnp.full(4, 0.2), jnp.zeros(0), jnp.asarray(0.0))[
        0
    ]
    np.testing.assert_allclose(residual[0], 0.0, atol=_EXACT)
    particle.bc.flux("c", node_indices=[0], fn=lambda q, t: q.missing)
    with pytest.raises(ds.LoweringError, match="q.missing.*not a field"):
        particle.build()


@pytest.mark.parametrize("cross_facet", [False, True])
def test_a_facet_flux_closes_its_total_and_a_traction_reads_the_facet_normal(cross_facet):
    """A facet flux named in ``flux_total`` closes the global so the flux rows sum to the
    target 0.8, and a traction ``k n`` on the top edge sums to ``(0, k)``."""
    problem = _plate(cross_facet)
    f, g = problem.physics_residual(
        problem.assemble_state({"c": 0.2, "u": 0.0}), jnp.array([0.5]), jnp.asarray(0.0)
    )
    np.testing.assert_allclose(g, 0.0, atol=_EXACT)
    np.testing.assert_allclose(jnp.sum(f[problem.field_slice("c")]), 0.8, atol=_EXACT)
    np.testing.assert_allclose(
        f[problem.field_slice("u")].reshape(-1, 2).sum(axis=0), [0.0, 2.0], atol=_EXACT
    )


def test_a_member_flux_reads_member_and_problem_parameters():
    """A family flux ``factor * q.m.k * (1 - c)`` scales each member's surface row by its own
    ``k`` and the problem's ``factor``."""
    spec = ds.problem("member_flux").domain("rod", ds.cartesian_grid_1d(2, 0.0, 1.0, method="fvm"))
    spec.field("u", mass="differential", scale=1.0)
    spec.param("factor", value=2.0, learnable=True)
    spec.equation(writes="u", reads=("u",))(lambda q: 0 * q.u)
    grid = ds.domains.spherical_grid_1d(3, 1.0, method="fvm")
    members = spec.family(
        "particles",
        count=2,
        domain=grid,
        params={"k": [2.0, 3.0]},
        scale={"c_s": 1.0},
        residual_scale={"c_s": 1.0},
    )
    members.field_decl("c_s", mass="differential")
    members.equation(writes="c_s", reads=("c_s",))(lambda q, m: 0 * q.c_s)
    members.bc.flux("c_s", node_indices=[-1], fn=lambda q, t: q.p.factor * q.m.k * (1 - q.c_s))
    problem = spec.build()
    f, _ = problem.physics_residual(
        jnp.full(problem.state_size(), 0.5), jnp.zeros(0), jnp.asarray(0.0)
    )
    expected = (
        jnp.zeros((2, 3))
        .at[:, -1]
        .set(jnp.array([2.0, 3.0]) * grid.face_areas[-1] / grid.cell_volumes[-1])
    )
    np.testing.assert_allclose(
        problem.family_view("particles").field(f, "c_s"), expected, atol=_EXACT
    )


def _flux_rows(cells):
    """Residual of ``laplacian(T) = 0`` with a unit inflow on the right edge of a two-triangle
    unit square, at ``T = x + 2 y``."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    mesh = ds.domains.TaggedMesh(
        points,
        np.asarray(cells),
        ele_type="TRI3",
        boundary_edges=np.array([[1, 3]]),
        boundary_tags=np.array([2]),
        tag_names={2: "right"},
    )
    spec = ds.problem("square_inflow").domain("square", mesh)
    spec.field("T", mass="differential", scale=1.0)
    spec.equation(writes="T", reads=("T",))(lambda q: q.ops.laplacian(q.T))
    spec.bc.flux("T", tag="right", value=1.0)
    state = spec.assemble_state({"T": jnp.asarray(points[:, 0] + 2.0 * points[:, 1])})
    return spec.build().physics_residual(state, jnp.zeros(0), jnp.asarray(0.0))[0]


def test_a_flux_on_a_clockwise_mesh_equals_the_flux_on_its_mirror():
    """Listing every triangle clockwise leaves the residual of a flux condition unchanged:
    the interior rows keep their sign relative to the boundary inflow."""
    np.testing.assert_allclose(
        _flux_rows([[0, 2, 1], [1, 2, 3]]), _flux_rows([[0, 1, 2], [1, 3, 2]]), atol=_EXACT
    )
