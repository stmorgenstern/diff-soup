r"""Operators: grid stencils on polynomials, curl on grids and meshes, divergence, and SI control
units through the public residual.

The curl expectations are read off rigid rotations, whose curl is known from the algebra
alone. The 2-D vortex ``u = (-2y, 3x)`` has scalar curl ``3 - (-2) = 5`` everywhere; the 3-D
rotation ``u = w x r`` with ``w = (1, 2, 3)`` has curl ``2w = (2, 4, 6)``. Both fields are
linear, so differences on a grid and gradients in a linear element reproduce them exactly. A
grid answers pointwise; a mesh answers the weak form ``\int (curl u) N_i``, which for a
constant curl ``k`` is ``k`` times the lumped mass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.domains import (
    Operators,
    box_mesh,
    cartesian_grid_1d,
    cartesian_grid_2d,
    cartesian_grid_3d,
    rectangle_mesh,
    staggered_grid_2d,
)

# measured 0.0 on the coefficient-length and traced-first checks, workstation CPU 2026-10-28,
# run one by one, and on the three composition rows, workstation CPU 2026-10-28, pytest
# tests/public_api/test_operators.py -k composes;
# measured at most 8.9e-16 on every stencil, rotation, divergence and residual-row check below
# (0.0 on the control-unit rows), CPU VM 2026-10-22, pytest tests/public_api/test_operators.py;
# measured at most 8.9e-16 on the transport checks (8.9e-16 on central, 1.1e-16 on the upwind
# inflow balance, 0.0 on the rest), workstation CPU 2026-10-28, pytest
# tests/public_api/test_operators.py -k grid_transport
_EXACT = 1e-12
_ROTATION = np.array([1.0, 2.0, 3.0])


def _vortex(coords):
    return jnp.asarray(np.stack([-2.0 * coords[:, 1], 3.0 * coords[:, 0]], axis=-1))


def _rotation(coords):
    return jnp.asarray(np.cross(np.broadcast_to(_ROTATION, coords.shape), coords))


@pytest.mark.parametrize("method", ["fvm", "fdm"])
def test_grid_stencils_are_exact_on_low_order_polynomials(method):
    """The interior laplacian of ``x (1 - x)`` is ``-2``, and the divergence of
    ``(a x, b y, c z)`` is ``a + b + c`` at every cell of a 2-D and a 3-D grid."""
    grid = cartesian_grid_1d(32, 0.0, 1.0, method=method)
    laplacian = Operators(grid).laplacian(grid.nodes * (1.0 - grid.nodes))
    np.testing.assert_allclose(laplacian[1:-1], -2.0, rtol=0.0, atol=_EXACT)
    coefficients = np.array([2.0, -3.0, 0.5])
    for grid in (
        cartesian_grid_2d(5, 4, 1.0, 0.8, method=method),
        cartesian_grid_3d(4, 3, 3, 1.0, 0.75, 1.5, method=method),
    ):
        coords = np.asarray(grid.node_coordinates())
        divergence = Operators(grid).divergence(jnp.asarray(coords * coefficients[: grid.dim]))
        np.testing.assert_allclose(divergence, coefficients[: grid.dim].sum(), atol=_EXACT)


def test_the_operator_bank_composes_with_jit_ad_vmap_and_keeps_the_field_dtype():
    """The laplacian runs under jit and vmap, differentiates, and keeps a float32 field
    float32 on a float64 grid; the grid and the mesh curl are linear under jit, vmap and jvp,
    and a fresh mesh traced first still runs eagerly."""
    grid = cartesian_grid_1d(12, 0.0, 1.0, method="fvm")
    ops = Operators(grid)

    def apply(values):
        return ops.laplacian(values)

    values = jnp.sin(jnp.pi * grid.nodes)
    np.testing.assert_allclose(jax.jit(apply)(values), apply(values), rtol=0.0, atol=_EXACT)
    batched = jax.vmap(apply)(jnp.stack((values, 2.0 * values)))
    np.testing.assert_allclose(batched[1], 2.0 * apply(values), rtol=0.0, atol=_EXACT)
    gradient = jax.grad(lambda scale: jnp.sum(apply(scale * values) ** 2))(1.0)
    np.testing.assert_allclose(gradient, 2.0 * jnp.sum(apply(values) ** 2), rtol=_EXACT)
    assert apply(values.astype(jnp.float32)).dtype == jnp.float32
    for domain in (
        cartesian_grid_2d(3, 3, 1.0, 1.0, method="fvm"),
        rectangle_mesh(2, 2, 1.0, 1.0, ele_type="TRI3"),
    ):
        bank = Operators(domain)
        curl = jax.jit(lambda u, bank=bank: bank.curl(u))
        field = _vortex(np.asarray(domain.node_coordinates()))
        batched = jax.vmap(curl)(jnp.stack((field, 2.0 * field)))
        np.testing.assert_allclose(bank.curl(field), batched[0], rtol=_EXACT, atol=_EXACT)
        np.testing.assert_allclose(batched[1], 2.0 * curl(field), rtol=_EXACT, atol=_EXACT)
        np.testing.assert_allclose(jax.jvp(curl, (field,), (field,))[1], curl(field), atol=_EXACT)


@pytest.mark.parametrize(
    "kind", ("grid-2d", "grid-3d", "mesh-2d", "mesh-3d"), ids=lambda kind: kind
)
def test_curl_of_a_rigid_rotation_is_twice_its_angular_velocity(kind):
    """A grid returns the curl pointwise; a mesh returns its weak lift, the curl times the
    lumped mass, component by component."""
    domain = {
        "grid-2d": lambda: cartesian_grid_2d(6, 5, 1.2, 1.0, method="fvm"),
        "grid-3d": lambda: cartesian_grid_3d(4, 3, 3, 1.0, 0.75, 1.5, method="fdm"),
        "mesh-2d": lambda: rectangle_mesh(3, 3, 1.2, 1.0, ele_type="TRI3"),
        "mesh-3d": lambda: box_mesh(2, 2, 2, 1.0, 1.0, 1.0, ele_type="TET4"),
    }[kind]()
    ops = Operators(domain)
    coords = np.asarray(domain.node_coordinates())
    field, expected = (_vortex, 5.0) if kind.endswith("2d") else (_rotation, 2.0 * _ROTATION)
    weight = np.asarray(ops.lumped_mass()) if kind.startswith("mesh") else np.ones(len(coords))
    expected = weight * expected if kind.endswith("2d") else weight[:, None] * expected
    np.testing.assert_allclose(ops.curl(field(coords)), expected, rtol=_EXACT, atol=_EXACT)


def test_curl_refuses_what_it_cannot_compute():
    """Curl refuses a component count the domain cannot support, a staggered grid (naming its
    cell grid) and a periodic declaration, like its sibling operators."""
    grid = cartesian_grid_2d(4, 4, 1.0, 1.0, method="fvm")
    ops = Operators(grid)
    for values in (jnp.zeros((grid.num_nodes, 3)), jnp.zeros((grid.num_nodes,))):
        with pytest.raises(ValueError, match="curl"):
            ops.curl(values)
    staggered = staggered_grid_2d(4, 4, 1.0, 1.0)
    with pytest.raises(NotImplementedError, match="cell_grid"):
        Operators(staggered).curl(jnp.zeros((staggered.num_nodes, 2)))
    spec = ds.problem("periodic_curl").domain("plane", grid)
    spec.field("u", mass="differential", components=2, scale=1.0)
    spec.equation(writes="u", reads=("u",))(
        lambda q: q.ops.curl(q.u)[..., None] * jnp.zeros_like(q.u)
    )
    spec.bc.periodic("u")
    with pytest.raises(NotImplementedError, match="curl has no periodic form"):
        spec.build()


def _vorticity_rows(domain, name):
    """The ``w`` rows of the residual of ``w = curl u`` at the frozen vortex ``u``."""
    spec = ds.problem(name).domain("plane", domain)
    quad = {} if name == "grid_curl" else {"quad": False}
    spec.field("u", mass="differential", components=2, scale=1.0, **quad)
    spec.field("w", mass="differential", scale=1.0, **quad)
    spec.equation(writes="u", reads=("u",))(lambda q: 0.0 * q.u)
    spec.equation(writes="w", reads=("w", "u"))(lambda q: q.ops.curl(q.u))
    problem = spec.build()
    coords = np.asarray(domain.node_coordinates())
    state = spec.assemble_state({"u": _vortex(coords), "w": jnp.zeros((len(coords),))})
    residual, _ = problem.physics_residual(
        state, jnp.zeros((problem.constraint_dim(),)), jnp.asarray(0.0)
    )
    return np.asarray(residual)[problem.field_slice("w")]


@pytest.mark.parametrize("kind", ("grid", "mesh"))
def test_a_recipe_curl_is_pointwise_on_a_grid_and_already_integrated_on_a_mesh(kind):
    """In a recipe equation the grid curl reaches the residual pointwise (5 per cell) and the
    mesh curl is not lifted a second time (5 times the lumped mass per node)."""
    if kind == "grid":
        grid = cartesian_grid_2d(5, 4, 1.0, 0.8, method="fvm")
        np.testing.assert_allclose(_vorticity_rows(grid, "grid_curl"), 5.0, atol=_EXACT)
    else:
        mesh = rectangle_mesh(3, 3, 1.2, 1.0, ele_type="TRI3")
        expected = 5.0 * np.asarray(Operators(mesh).lumped_mass())
        np.testing.assert_allclose(_vorticity_rows(mesh, "mesh_curl"), expected, atol=_EXACT)


def _control_recipe(*, time_scale=None, global_scale=None, global_offset=None):
    """Two cells, ``du/dt = -u + drive``, and ``protocol(t) = 1 + t`` bound to ``drive``."""
    spec = ds.problem("control_units").domain("line", cartesian_grid_1d(2, 0.0, 1.0, method="fvm"))
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.zeros_like(x))
    spec.global_var("drive", initial=1.0, scale=global_scale)
    spec.equation(writes="u", reads=("u", "drive"))(lambda q: -q.u + q["drive"])
    spec.constraint.domain_integral(
        name="pin",
        field="u",
        integrand_fn=lambda q: q.u,
        target_fn=lambda t: jnp.asarray(0.0),
        closes="drive",
    )
    spec.control.derived(name="protocol", fn=lambda t: 1.0 + t, binds_to="drive")
    if time_scale is not None:
        spec.time_scale(time_scale)
    if global_offset is not None:
        spec.scales(state_offset={"drive": global_offset})
    return spec


@pytest.mark.parametrize(
    "scales,control,target",
    (
        ({"time_scale": 4.0}, None, 2.0),
        ({"global_scale": 2.0}, None, 0.625),
        ({"global_scale": 2.0, "global_offset": 0.5}, None, 0.375),
        ({"global_scale": 2.0}, {"protocol": 5.0}, 2.5),
        ({}, None, 1.25),
    ),
    ids=("si-time", "si-value", "offset", "si-payload", "identity"),
)
def test_a_bound_control_reads_si_time_and_lands_in_internal_units(scales, control, target):
    """At internal ``t = 0.25`` and ``drive = 3`` the closing row is ``drive - target``:
    ``time_ref`` scales the time ``fn`` reads, and the SI value reaches the multiplier as
    ``(value - offset) / scale``; the state rows ``-(-u + drive)`` are untouched.

    Worked targets: ``fn(4 x 0.25) = 2``; ``1.25 / 2``; ``(1.25 - 0.5) / 2``; ``5 / 2``;
    ``1.25``."""
    problem = _control_recipe(**scales).build()
    state = jnp.asarray([0.4, 0.9, 3.0])
    residual = problem.solver_problem().residual(
        jnp.asarray(0.25), state, jnp.zeros_like(state), problem.default_theta(), control
    )
    np.testing.assert_allclose(residual, [-2.6, -2.1, 3.0 - target], rtol=0.0, atol=_EXACT)


def test_the_lowered_controls_surface_takes_internal_time():
    """``LoweredProblem.controls`` at internal ``t = 0.25`` with ``time_ref = 4`` evaluates
    ``fn`` at SI ``t = 1``."""
    values = _control_recipe(time_scale=4.0).build().controls(jnp.asarray(0.25))
    np.testing.assert_allclose(values["protocol"], 2.0, rtol=0.0, atol=_EXACT)


def _per_cell_coefficient():
    mesh = rectangle_mesh(3, 2, 1.0, 1.0)
    ops = Operators(mesh)
    coords = np.asarray(mesh.node_coordinates())
    u = jnp.asarray(coords[:, 0] ** 2 + coords[:, 1])
    n_cells, n_q = ops.interp_to_quad(u).shape
    return ops, u, jnp.linspace(1.0, 2.0, n_cells), n_q


@pytest.mark.parametrize("operator", ["div_flux", "mass", "wrong length"])
def test_a_one_dimensional_coefficient_is_read_by_its_length(operator):
    """On a mesh a coefficient with one value per cell equals the same values repeated at each
    cell's quadrature points, and one with neither a value per node nor a value per cell is
    refused."""
    ops, u, per_cell, n_q = _per_cell_coefficient()
    if operator == "wrong length":
        with pytest.raises(ValueError, match="num_dofs=12.*n_cells=6"):
            ops.div_flux(per_cell[:-1], u)
        return
    apply = (lambda D: ops.div_flux(D, u)) if operator == "div_flux" else ops.mass
    at_quad = jnp.repeat(per_cell[:, None], n_q, axis=1)
    np.testing.assert_allclose(apply(per_cell), apply(at_quad), rtol=0.0, atol=_EXACT)


_NODAL = 0.25 + np.arange(16) / 16.0


def _explicit_composition(q):
    """``jnp.where(abs(T) > 0.5, T**2, nodal - T) + T * lap`` with explicit arrays."""
    at_quad = q.ops.interp_to_quad(q.ops.unwrap_field(q.T))
    nodal = q.ops.interp_to_quad(jnp.asarray(_NODAL))
    pointwise = jnp.where(jnp.abs(at_quad) > 0.5, at_quad * at_quad, nodal - at_quad)
    return pointwise + q.ops.unwrap_field(q.T) * q.ops.laplacian(q.T)


_COMPOSED = {
    "1 + lap": (
        lambda q: 1.0 + q.ops.laplacian(q.T),
        lambda q: q.ops.laplacian(q.T) + 1.0,
        lambda q: q.ops.laplacian(q.T) + jnp.ones(16),
    ),
    "constant": (lambda q: 1.0, lambda q: jnp.ones(16)),
    "field algebra": (
        lambda q: jnp.where(abs(q.T) > 0.5, q.T**2, _NODAL - q.T) + q.T * q.ops.laplacian(q.T),
        _explicit_composition,
    ),
}


@pytest.mark.parametrize("form", list(_COMPOSED), ids=list(_COMPOSED))
def test_a_mesh_field_composes_like_its_explicit_form(form):
    """On a mesh a constant joins a residual as the source ``1`` would, on either side of an
    operator output, and a field read composes with ``abs``, comparisons, powers, nodal arrays
    and operator outputs: each residual equals the one written with explicit nodal or
    quadrature arrays."""
    mesh = rectangle_mesh(3, 3, 1.0, 1.0, ele_type="TRI3")
    coords = np.asarray(mesh.node_coordinates())
    rows = []
    for index, expr in enumerate(_COMPOSED[form]):
        spec = ds.problem(f"composed_{index}").domain("plate", mesh)
        spec.field("T", mass="differential", scale=1.0)
        spec.equation(writes="T", reads=("T",))(expr)
        state = spec.assemble_state({"T": jnp.asarray(0.2 + coords[:, 0] - coords[:, 1] ** 2)})
        rows.append(spec.build().physics_residual(state, jnp.zeros(0), jnp.asarray(0.0))[0])
    for row in rows[:-1]:
        np.testing.assert_allclose(row, rows[-1], rtol=0.0, atol=_EXACT)


def _bed_cells(method):
    grid = cartesian_grid_2d(6, 3, 1.2, 0.6, method=method)
    coords = np.asarray(grid.node_coordinates())
    field = jnp.asarray(np.sin(3.0 * coords[:, 0]) + coords[:, 0] * coords[:, 1] ** 2)
    return grid, coords, field


@pytest.mark.parametrize("method", ["fvm", "fdm"])
@pytest.mark.parametrize(
    "claim",
    ["equal axes", "no y", "velocity field", "central", "inflow balance", "upwind inflow balance"],
)
def test_grid_transport_takes_per_axis_dispersion_velocity_fields_and_an_inflow(method, claim):
    """``div_flux`` with the same cell diffusivity on both axes equals the shared one, and with
    ``(1.5, 0)`` the 1-D operator along x; ``advect_upwind`` with velocity arrays equal to constants equals the
    constant form; ``div_advect`` by default gives each interior face the mean of ``v c`` in its
    two cells, the paper's ``(v c[i+1] - v c[i-1]) / 2h`` for a velocity field on a 3-D grid,
    and refuses another scheme; and with ``-div_advect`` under either scheme and an inflow
    ``0.5 * 2`` on the low-x face, the rate of change of the integral is the inflow minus the
    outflow ``v c`` on the high-x face."""
    grid, coords, field = _bed_cells(method)
    ops = Operators(grid)
    if claim == "equal axes":
        D = jnp.asarray(1.0 + 0.5 * coords[:, 0])
        got, expected = ops.div_flux((D, D), field), ops.div_flux(D, field)
    elif claim == "no y":
        line = Operators(cartesian_grid_1d(6, 0.0, 1.2, method=method))
        got = ops.div_flux((1.5, 0.0), field).reshape(6, 3)
        expected = jax.vmap(lambda c: line.div_flux(1.5, c), in_axes=1, out_axes=1)(
            field.reshape(6, 3)
        )
    elif claim == "velocity field":
        uniform = (jnp.full(18, 0.3), jnp.full(18, -0.2))
        got, expected = ops.advect_upwind(field, uniform), ops.advect_upwind(field, (0.3, -0.2))
    elif claim == "central":
        box = cartesian_grid_3d(5, 3, 2, 1.0, 0.6, 0.4, method=method)
        xyz = np.asarray(box.node_coordinates())
        u = jnp.asarray(1.0 + xyz[:, 0] * xyz[:, 1])
        c = jnp.asarray(np.sin(3.0 * xyz[:, 0]) + xyz[:, 2])
        with pytest.raises(ValueError, match='"central" or "upwind"'):
            Operators(box).div_advect(c, (u, 0.0, 0.0), scheme="quick")
        got = Operators(box).div_advect(c, (u, 0.0, 0.0)).reshape(5, 3, 2)[1:-1]
        uc = (u * c).reshape(5, 3, 2)
        expected = (uc[2:] - uc[:-2]) / (2 * np.diff(np.unique(xyz[:, 0]))[0])
    else:
        kw = {"scheme": "upwind"} if claim == "upwind inflow balance" else {}
        speed = jnp.asarray(0.5 + 0.1 * coords[:, 1])
        spec = ds.problem(f"inflow_{method}").domain("bed", grid)
        spec.field("c", mass="differential", scale=1.0)
        spec.equation(writes="c", reads=("c",))(
            lambda q: -q.ops.div_advect(q.c, (speed, 0.0), **kw)
        )
        spec.bc.inflow("c", axis=0, side="low", velocity=0.5, value=2.0)
        rows = spec.build().physics_residual(
            spec.assemble_state({"c": field}), jnp.zeros(0), jnp.asarray(0.0)
        )[0]
        got = jnp.sum(rows * jnp.asarray(grid.cell_volumes))
        outlet = coords[:, 0] == coords[:, 0].max()
        expected = 0.5 * 2.0 * 0.6 - jnp.sum((speed * field)[outlet]) * 0.2
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=_EXACT)
