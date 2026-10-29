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
# on the residual rows of test_a_flux_on_a_clockwise_mesh_equals_the_flux_on_its_mirror and
# 5.6e-17 on its lumped masses, workstation CPU 2026-10-28;
# measured 0.0 on the injected totals and the mesh rows of
# test_a_grid_flux_is_weighted_by_the_one_face_its_cells_lie_on, workstation CPU 2026-10-28
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
    the interior rows keep their sign relative to the boundary inflow, and each node keeps its
    positive lumped mass."""
    clockwise, counter = [[0, 2, 1], [1, 2, 3]], [[0, 1, 2], [1, 3, 2]]
    np.testing.assert_allclose(_flux_rows(clockwise), _flux_rows(counter), atol=_EXACT)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    masses = [
        ds.domains.Operators(
            ds.domains.Mesh(points, np.array(cells), ele_type="TRI3")
        ).lumped_mass()
        for cells in (clockwise, counter)
    ]
    np.testing.assert_allclose(masses[0], masses[1], rtol=0.0, atol=_EXACT)


def _grid_spec(name, grid, equation):
    spec = ds.problem(name).domain("grid", grid)
    spec.field("u", mass="differential", scale=1.0)
    spec.equation(writes="u", reads=("u",))(equation)
    return spec


def _rows(spec, values=None):
    state = spec.assemble_state(values or {})
    return np.asarray(spec.build().physics_residual(state, jnp.zeros(0), jnp.asarray(0.0))[0])


@pytest.mark.parametrize("selection", ["axis-side", "cells", "corner", "mesh nodes"])
def test_a_grid_flux_is_weighted_by_the_one_face_its_cells_lie_on(selection):
    """A unit flux on the low-y face of a 2 x 1.5 grid injects the face length, 2, whether the
    face is named by axis and side or by its cells; a lone corner cell, which lies on two faces,
    is refused. The ``"mesh nodes"`` row is a mesh claim: there a flux by ``node_indices``
    selects the boundary facets of those nodes, so the left side's nodes of a quadrilateral
    mesh give the residual of ``tag="left"``."""
    if selection == "mesh nodes":
        mesh = ds.domains.rectangle_mesh(3, 2, 2.0, 1.0, ele_type="QUAD4", with_boundary_tags=True)
        x = np.asarray(mesh.node_coordinates())[:, 0]
        rows = []
        for where in ({"tag": "left"}, {"node_indices": np.flatnonzero(x == 0.0)}):
            spec = _grid_spec("mesh_flux", mesh, lambda q: q.ops.laplacian(q.u))
            spec.bc.flux("u", value=1.0, **where)
            rows.append(_rows(spec, {"u": jnp.asarray(x**2)}))
        np.testing.assert_allclose(rows[1], rows[0], atol=_EXACT)
        return
    grid = ds.domains.cartesian_grid_2d(4, 3, 2.0, 1.5, method="fvm")
    spec = _grid_spec(f"face_flux_{selection}", grid, lambda q: q.ops.laplacian(q.u))
    if selection == "axis-side":
        spec.bc.flux("u", axis=1, side="low", value=1.0)
    else:
        cells = [0, 3, 6, 9] if selection == "cells" else [0]
        spec.bc.flux("u", node_indices=cells, value=1.0)
    if selection == "corner":
        with pytest.raises(ValueError, match="axis=0, side='low'; axis=1, side='low'"):
            spec.build()
        return
    injected = np.sum(_rows(spec) * np.asarray(grid.cell_volumes))
    np.testing.assert_allclose(injected, 2.0, rtol=_EXACT)


@pytest.mark.parametrize("method", ["fvm", "fdm", "fem"])
def test_neumann_fixes_the_normal_derivative_whatever_the_diffusivity(method):
    """With ``div(2 grad u) = 0``, ``u = 0`` on the low-x cells (or nodes) and ``du/dn = 0.7``
    on the high-x side, the profile ``u = 0.7 (x - x_0)`` is steady: every residual row is
    zero. Imposing the flux ``2 du/dn = 0.7`` instead would halve the slope."""
    if method == "fem":
        grid = ds.domains.rectangle_mesh(4, 2, 1.0, 0.5, ele_type="TRI3", with_boundary_tags=True)
    else:
        grid = ds.domains.cartesian_grid_2d(8, 3, 1.0, 0.5, method=method)
    x = np.asarray(grid.node_coordinates())[:, 0]
    spec = _grid_spec(f"neumann_{method}", grid, lambda q: q.ops.div_flux(2.0, q.u))
    spec.bc.dirichlet("u", node_indices=np.flatnonzero(x == x.min()), value=0.0)
    if method == "fem":
        spec.bc.neumann("u", tag="right", value=0.7)
    else:
        spec.bc.neumann("u", axis=0, side="high", value=0.7)
    rows = _rows(spec, {"u": jnp.asarray(0.7 * (x - x.min()))})
    # measured at most 1.4e-14 over the three rows, workstation CPU 2026-10-28,
    # pytest tests/public_api/test_boundary_callbacks.py -k neumann
    np.testing.assert_allclose(rows, 0.0, atol=1e-12)


@pytest.mark.parametrize("method,shape", [("fvm", (12, 9)), ("fdm", (12, 9)), ("fdm", (8, 6, 5))])
def test_a_field_periodic_on_one_axis_takes_dirichlet_walls_on_the_others(method, shape):
    """Periodic in x with ``u = 0`` on the cells of every other face, the mode
    ``sin(2 pi x) prod sin(pi (y - y_0) / L)`` is an eigenvector of the laplacian: each free row
    is the exact semi-discrete decay rate times ``u``."""
    extent = (1.0, 0.8, 0.6)[: len(shape)]
    grid = (ds.domains.cartesian_grid_2d if len(shape) == 2 else ds.domains.cartesian_grid_3d)(
        *shape, *extent, method=method
    )
    coords = np.asarray(grid.node_coordinates())
    spec = _grid_spec(f"periodic_x_{method}_{len(shape)}", grid, lambda q: q.ops.laplacian(q.u))
    spec.bc.periodic("u", axes=(0,))
    h = np.asarray(extent) / np.asarray(shape)
    mode = np.sin(2.0 * np.pi * coords[:, 0])
    rate = -4.0 / h[0] ** 2 * np.sin(np.pi * h[0]) ** 2
    walls = np.zeros(len(coords), dtype=bool)
    for d in range(1, len(shape)):
        span = extent[d] - h[d]
        mode = mode * np.sin(np.pi * (coords[:, d] - h[d] / 2) / span)
        rate -= 4.0 / h[d] ** 2 * np.sin(np.pi * h[d] / (2.0 * span)) ** 2
        walls |= np.isclose(coords[:, d], h[d] / 2) | np.isclose(coords[:, d], extent[d] - h[d] / 2)
    spec.bc.dirichlet("u", node_indices=np.flatnonzero(walls), value=0.0)
    rows = _rows(spec, {"u": jnp.asarray(mode)})
    # measured at most 2.0e-13 on the free rows (rate about -58 and -100), workstation CPU
    # 2026-10-28, pytest tests/public_api/test_boundary_callbacks.py -k periodic
    np.testing.assert_allclose(rows[~walls], rate * mode[~walls], atol=1e-11)
