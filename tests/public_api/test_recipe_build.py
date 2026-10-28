"""Recipe API: domains and the grid factory, fields and equations, custom operators and
transfers, couplings and read-through aliases, build identity, globals and operational
control tables."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.numpy import where

import diff_soup as ds
from diff_soup.domains import (
    Operators,
    TaggedMesh,
    box_mesh,
    cartesian_grid_2d,
    interval_mesh,
    rectangle_mesh,
)
from diff_soup.utils import (
    build_padded_color_arrays,
    build_sparse_jvp_groups,
    sparse_jvp_jacobian_values,
)
from tests.support.problems import operational_cell

# measured 0.0 on every residual, Jacobian and grid-geometry comparison below, CPU VM
# 2026-10-22, pytest tests/public_api/test_recipe_build.py
_EXACT = 1e-12


def _residual(problem, state, multipliers=None):
    lam = jnp.zeros((0,), state.dtype) if multipliers is None else multipliers
    return problem.physics_residual(state, lam, jnp.asarray(0.0, state.dtype))[0]


def test_the_grid_factory_places_nodes_at_cell_centres_of_given_faces():
    """``ds.cartesian_grid_1d(faces=...)`` puts nodes at the cell centres with the face
    spacing as cell widths, its laplacian of ``x^2`` is 2 on the uniform stretches, and
    non-increasing faces or mixed forms are refused; on every grid ``coordinates`` holds the
    per-axis centres and ``meshgrid()`` the coordinates in field storage order."""
    faces = np.concatenate(
        [np.linspace(0.0, 0.4, 5), [0.425, 0.45, 0.475, 0.5], np.linspace(0.6, 1.0, 5)]
    )
    grid = ds.cartesian_grid_1d(faces=faces)
    centres = 0.5 * (faces[:-1] + faces[1:])
    np.testing.assert_allclose(grid.cell_widths, np.diff(faces), rtol=_EXACT)
    np.testing.assert_allclose(grid.nodes, centres, rtol=_EXACT)
    laplacian = np.asarray(Operators(grid).laplacian(grid.nodes**2))
    # measured 3.3e-15 relative on the uniform interior cells, CPU VM 2026-10-22,
    # pytest tests/public_api/test_recipe_build.py::test_the_grid_factory_places_nodes_at_cell_centres_of_given_faces
    np.testing.assert_allclose(laplacian[[1, 2, 5, 6, 9, 10, 11]], 2.0, rtol=1e-9)
    with pytest.raises(ValueError, match="strictly increasing"):
        ds.cartesian_grid_1d(faces=faces[::-1])
    with pytest.raises(ValueError):
        ds.cartesian_grid_1d(4, 0.0, 1.0, faces=faces)
    plate = cartesian_grid_2d(3, 4, 1.0, 2.0)
    x, y = plate.meshgrid()
    assert x.shape == y.shape == plate.shape
    np.testing.assert_array_equal(np.stack((x.ravel(), y.ravel()), -1), plate.node_coordinates())
    for coordinate, axis in zip(plate.coordinates, plate.axes, strict=True):
        np.testing.assert_array_equal(coordinate, axis.nodes)
    np.testing.assert_array_equal(grid.meshgrid()[0], grid.coordinates[0])
    np.testing.assert_array_equal(grid.coordinates[0], grid.nodes)


def _decay_spec(name, domain, operators=None, init=None):
    spec = ds.problem(name).domain(
        "part", domain, **({} if operators is None else {"operators": operators})
    )
    spec.field("u", mass="differential", scale=1.0, init=init)
    spec.equation(writes="u", reads=("u",))(lambda q: q.ops.laplacian(q.u) - 0.4 * q.u)
    return spec


def test_a_built_recipe_is_cached_and_its_residual_transforms():
    """``build()`` returns one problem object per recipe, on a grid and on a mesh, and its
    residual runs under jit, grad and vmap; a template's ``spec(...)`` renders the ``init=``
    that its built problem's ``assemble_state`` ignores."""
    for domain in (ds.cartesian_grid_1d(5, 0.0, 1.0, method="fvm"), rectangle_mesh(2, 2, 1.0, 1.0)):
        spec = _decay_spec(f"cached_{type(domain).__name__}", domain)
        problem = spec.build()
        assert spec.build() is problem
        state = jnp.linspace(0.5, 1.5, problem.state_size())
        np.testing.assert_allclose(
            jax.jit(lambda v: _residual(problem, v))(state), _residual(problem, state), atol=_EXACT
        )
        gradient = jax.grad(lambda v: jnp.sum(_residual(problem, v) ** 2))(state)
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert jax.vmap(lambda v: _residual(problem, v))(jnp.stack((state, state))).shape == (
            2,
            state.size,
        )

    @ds.recipe
    def rod(n=5):
        return _decay_spec(f"rod_{n}", ds.cartesian_grid_1d(n, 0.0, 1.0), init=lambda x: 1.0 + x)

    nodes = ds.cartesian_grid_1d(4, 0.0, 1.0).nodes
    np.testing.assert_array_equal(rod.build(n=4).assemble_state({}), 0.0)
    np.testing.assert_allclose(rod.spec(n=4).assemble_state({}), 1.0 + nodes, rtol=_EXACT)


class _ScaledOperators(Operators):
    def laplacian(self, values, *, bcs=None, t=None):
        return 7.0 * values


def test_a_custom_operator_bank_is_the_one_the_lowered_equations_call():
    """``domain(..., operators=)`` replaces the bank the equations' ``q.ops`` reads."""
    spec = _decay_spec(
        "custom_operators", ds.cartesian_grid_1d(4, 0.0, 1.0, method="fvm"), _ScaledOperators
    )
    problem = spec.build()
    np.testing.assert_allclose(_residual(problem, jnp.ones(4)), 7.0 - 0.4, atol=_EXACT)


def _alias_recipe(name, bundled):
    spec = ds.problem(name).domain("line", ds.cartesian_grid_1d(4, 0.0, 1.0, method="fvm"))
    spec.domain("supply", ds.cartesian_grid_1d(4, 0.0, 1.0, method="fvm"))
    supply = spec.field(
        "s", domain="supply", mass="differential", scale=1.0, init=lambda x: 1.0 + x
    )
    for field in ("a", "b"):
        spec.field(
            field, domain="line", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x)
        )
    spec.equation(writes="s", reads=("s",))(lambda q: -q.s)
    spec.couple(supply, onto="line", as_="drive", at=(0, 1, 2, 3))
    if bundled:
        spec.equations(writes=("a", "b"), reads=("a", "b", "drive"))(
            lambda q: {"a": -q.a + q.drive, "b": -2.0 * q.b + q.drive}
        )
    else:
        spec.equation(writes="a", reads=("a", "drive"))(lambda q: -q.a + q.drive)
        spec.equation(writes="b", reads=("b", "drive"))(lambda q: -2.0 * q.b + q.drive)
    return spec


def test_a_multi_output_equation_reads_an_alias_like_split_equations_and_aliases_cannot_shadow():
    """A bundled two-output equation reading a ``couple(..., as_=)`` alias gives the split
    equations' residual and Jacobian; an alias that names a field of its component is refused."""
    spec = _alias_recipe("alias_bundled", True)
    bundled, split = spec.build(), _alias_recipe("alias_split", False).build()
    state = jnp.asarray(spec.assemble_state({}))
    np.testing.assert_allclose(_residual(bundled, state), _residual(split, state), atol=_EXACT)
    np.testing.assert_allclose(
        jax.jacfwd(lambda v: _residual(bundled, v))(state),
        jax.jacfwd(lambda v: _residual(split, v))(state),
        atol=_EXACT,
    )
    clash = ds.problem("alias_clash").domain(
        "line", ds.cartesian_grid_1d(4, 0.0, 1.0, method="fvm")
    )
    clash.domain("supply", ds.cartesian_grid_1d(4, 0.0, 1.0, method="fvm"))
    supply = clash.field("s", domain="supply", mass="differential", scale=1.0)
    clash.field("u", domain="line", mass="differential", scale=1.0)
    with pytest.raises(ds.ConfigurationError, match="'u'.*field in component 'line'"):
        clash.couple(supply, onto="line", at=(0,), as_="u")


class _EveryOtherNode(ds.TransferOp):
    def alias_size(self, *, n_source, n_target):
        return (n_source + 1) // 2

    def apply(self, source, target_size):
        return source[::2]

    def sparsity(self, source_size, target_size):
        return np.arange(target_size), np.arange(0, source_size, 2)


class _WeightedHead(ds.TransferOp):
    def alias_size(self, *, n_source, n_target):
        return n_target

    def apply(self, source, target_size):
        return jnp.full((target_size,), 0.25 * source[0] + 0.75 * source[1])

    def sparsity(self, source_size, target_size):
        return np.repeat(np.arange(target_size), 2), np.tile([0, 1], target_size)


@pytest.mark.parametrize("transfer", ("every-other-node", "weighted-head"))
def test_a_custom_transfer_couples_domains_as_its_hand_oracle(transfer):
    """A ``ds.TransferOp`` that subclasses nothing built in couples a four-cell supply into a
    two-cell line: every other node, or a bundled two-field kernel's weighted head."""
    spec = ds.problem(f"transfer_{transfer}").domain(
        "line", ds.cartesian_grid_1d(2, 0.0, 1.0, method="fvm")
    )
    spec.domain("supply", ds.cartesian_grid_1d(4, 0.0, 1.0, method="fvm"))
    c1 = spec.field("c1", domain="supply", mass="differential", scale=1.0)
    spec.field("c2", domain="supply", mass="differential", scale=1.0)
    spec.field("u", domain="line", mass="differential", scale=1.0)
    spec.equations(writes=("c1", "c2"), reads=("c1", "c2"))(
        lambda q: {"c1": -q.c1 + q.c1 * q.c2, "c2": -2.0 * q.c2 + q.c1 * q.c2}
    )
    op = _EveryOtherNode() if transfer == "every-other-node" else _WeightedHead()
    spec.couple(c1, onto="line", as_="moved", via=op)
    spec.equation(writes="u", reads=("u", "moved"))(lambda q: -q.u + q.moved)
    problem = spec.build()
    state = jnp.linspace(0.3, 1.7, problem.state_size())
    residual = _residual(problem, state)
    c1v, c2v, u = (state[problem.field_slice(name)] for name in ("c1", "c2", "u"))
    moved = c1v[::2] if transfer == "every-other-node" else 0.25 * c1v[0] + 0.75 * c1v[1]
    np.testing.assert_allclose(residual[problem.field_slice("u")], -u + moved, atol=_EXACT)
    np.testing.assert_allclose(residual[problem.field_slice("c1")], -c1v + c1v * c2v, atol=_EXACT)


def test_a_differential_global_is_authored_with_its_role_and_needs_exactly_one_closure():
    """A differential global appears in the layout with its role; one without an equation,
    or with an equation and a constraint, is refused at build naming it."""
    spec = ds.problem("global_roles").domain(
        "line", ds.cartesian_grid_1d(2, 0.0, 1.0, method="fvm")
    )
    spec.field("u", mass="differential", scale=1.0)
    spec.equation(writes="u", reads=("u",))(lambda q: -q.u)
    spec.global_var("T", initial=0.0, mass="differential")
    with pytest.raises(ValueError, match="T"):
        spec.build()
    spec.equation(writes="T", reads=("T",))(lambda q: 1.0 - q["T"])
    layout = {entry["name"]: entry for entry in spec.build().global_layout()}
    assert (layout["T"]["mass"], layout["T"]["size"]) == ("differential", 1)
    twice = ds.problem("global_twice").domain(
        "line", ds.cartesian_grid_1d(2, 0.0, 1.0, method="fvm")
    )
    twice.field("u", mass="differential", scale=1.0)
    twice.equation(writes="u", reads=("u",))(lambda q: -q.u)
    twice.global_var("T", initial=0.0, mass="differential")
    twice.equation(writes="T", reads=("T",))(lambda q: -q["T"])
    twice.constraint.galvanostatic(
        name="double_close",
        closes="T",
        current_fn=lambda q: jnp.mean(q.u),
        target_fn=lambda t: jnp.asarray(1.0),
    )
    with pytest.raises(ValueError, match="T"):
        twice.build()


def test_an_operational_control_table_lowers_and_invalid_tables_are_refused():
    """The table lowers to one OperationalControlConstraint closing the current unknown;
    duplicate modes, an unknown voltage, a bad threshold and a second table are refused."""
    equations = operational_cell().to_equation_set()
    (table,) = [c for c in equations.constraints if isinstance(c, ds.OperationalControlConstraint)]
    assert (table.name, table.closes.name, table.modes) == (
        "terminal",
        "I_cell",
        ("current", "voltage", "power"),
    )
    with pytest.raises(ValueError, match="[Dd]uplicate|once"):
        operational_cell(modes=("current", "current"))
    with pytest.raises(ValueError, match="missing_voltage"):
        operational_cell(voltage="missing_voltage").build()
    for threshold in (-1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="seed_voltage_threshold"):
            operational_cell(seed_voltage_threshold=threshold)
    with pytest.raises(TypeError, match="seed_voltage_threshold"):
        operational_cell(seed_voltage_threshold="1e-6")
    second = operational_cell()
    second.control.operational(
        name="second",
        current="terminal_current",
        voltage="terminal_voltage",
        current_unknown="I_cell",
        modes=("current", "voltage"),
    )
    with pytest.raises(ValueError, match="only one operational control table"):
        second.build()


def test_a_recipe_whose_equation_calls_a_name_imported_function_builds_and_solves():
    """A recipe whose equation calls ``where`` imported by name from ``jax.numpy`` builds, and
    ``u' = -where(u > 0, rate u, 0)`` with ``rate = 0.4`` from ``u = 1`` solves to
    ``exp(-0.4 t)``."""
    spec = ds.problem("name_imported_where").domain(
        "line", ds.cartesian_grid_1d(3, 0.0, 1.0, method="fvm")
    )
    spec.param("rate", value=0.4, learnable=False)
    spec.field("u", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
    spec.equation(writes="u", reads=("u",))(lambda q: -where(q.u > 0.0, q.p.rate * q.u, 0.0))
    problem = spec.build()
    solution = ds.pdesolve(
        problem,
        ds.JNDF.stock(linear="dense", **{"rtol": 1e-8, "atol": 1e-10}),
        (0.0, 1.0),
        jnp.ones(problem.state_size()),
    )
    # measured 1.8e-8 relative on every cell, CPU VM 2026-10-23,
    # pytest tests/public_api/test_recipe_build.py::test_a_recipe_whose_equation_calls_a_name_imported_function_builds_and_solves
    np.testing.assert_allclose(solution.ys[-1], np.exp(-0.4), rtol=1e-6)


def _diffusion_line(with_global):
    """``u' = 0.05 u_xx`` on six cells; with a global ``load`` read by every row and closed by
    ``load = mean(u)``."""
    spec = ds.problem(f"topology_line_{with_global}").domain(
        "line", ds.cartesian_grid_1d(6, 0.0, 1.0, method="fvm")
    )
    spec.field("u", mass="differential", scale=1.0, init=lambda x: 1.0 + x)
    if not with_global:
        spec.equation(writes="u", reads=("u",))(lambda q: 0.05 * q.ops.laplacian(q.u))
        return spec.build()
    spec.global_var("load", initial=0.0, scale=1.0)
    spec.equation(writes="u", reads=("u", "load"))(
        lambda q: 0.05 * q.ops.laplacian(q.u) + q["load"]
    )
    spec.constraint.galvanostatic(
        name="load_closure",
        closes="load",
        current_fn=lambda q: q["load"] - jnp.mean(q.u),
        target_fn=lambda time: jnp.asarray(0.0),
    )
    return spec.build()


def test_the_topology_report_attributes_a_dense_global_row_and_counts_its_colors():
    """The six-cell diffusion stencil is tridiagonal (16 entries, 3 colors); a global read by
    every row and closing on the mean adds a full row and column: the widest row is the
    global's, with degree 7, the entries split 22 to ``u`` and 7 to ``load``, and the full row
    couples every column, so the coloring needs one color per unknown."""
    stencil = ds.topology_report(_diffusion_line(False))
    assert (stencil.nnz, stencil.color_count, stencil.per_variable_nnz) == (16, 3, {"u": 16})
    report = ds.topology_report(_diffusion_line(True), top_k=1)
    assert (report.n, report.nnz, report.color_count) == (7, 29, 7)
    assert report.widest_rows == ((6, 7, "load"),)
    assert report.per_variable_nnz == {"u": 22, "load": 7}


def test_sparse_jvp_groups_recover_a_tridiagonal_jacobian_exactly():
    """build_sparse_jvp_groups colors a six-column tridiagonal pattern with three colors, and
    the padded groups recover through sparse_jvp_jacobian_values exactly the entries
    ``jax.jacfwd`` gives."""
    index = np.arange(6)
    rows = np.concatenate([index, index[1:], index[:-1]])
    cols = np.concatenate([index, index[:-1], index[1:]])
    groups = build_sparse_jvp_groups(rows, cols, 6)
    assert sorted(groups[0]) == [(0, 3), (1, 4), (2, 5)]

    def residual(u):
        near = jnp.zeros_like(u).at[1:].add(u[:-1]).at[:-1].add(u[1:])
        return u**3 - 0.5 * near

    state = jnp.linspace(0.2, 1.0, 6)
    values = sparse_jvp_jacobian_values(
        residual, state, *build_padded_color_arrays(*groups), nnz=rows.size
    )
    np.testing.assert_array_equal(values, np.asarray(jax.jacfwd(residual)(state))[rows, cols])


def _heat_end_state(domain, residual_form, coeff, condition=None, reaction=None):
    """``T`` at ``t = 0.5`` of ``coeff T' = lap(T) + 1 - k T`` from ``sin(pi x) + 1``, written as
    the residual ``coeff q.dt(T) - lap(T) - 1 + k q.T`` or as the right-hand side
    ``(lap(T) + 1 - k q.T) / coeff``, with an optional reaction rate ``k`` and an optional
    condition at the last node."""
    label = coeff if np.ndim(coeff) == 0 else f"array{np.size(coeff)}"
    spec = ds.problem(f"heat_{residual_form}_{label}_{condition}_{reaction}").domain("d", domain)
    spec.field("T", mass="differential", scale=1.0, init=lambda x, *_: jnp.sin(jnp.pi * x) + 1.0)
    if residual_form and reaction is None:
        rhs = lambda q: coeff * q.dt("T") - q.ops.laplacian(q.T) - 1.0  # noqa: E731
    elif residual_form:
        rhs = lambda q: coeff * q.dt("T") - q.ops.laplacian(q.T) - 1.0 + reaction * q.T  # noqa: E731
    elif reaction is None:
        rhs = lambda q: (q.ops.laplacian(q.T) + 1.0) / coeff  # noqa: E731
    else:
        rhs = lambda q: (q.ops.laplacian(q.T) + 1.0 - reaction * q.T) / coeff  # noqa: E731
    spec.equation(writes="T", reads=("T",))(rhs)
    last = np.array([domain.num_nodes - 1])
    if condition == "flux":
        spec.bc.flux("T", node_indices=last, value=1.0)
    elif condition == "robin":
        spec.bc.robin("T", node_indices=last, alpha=1.0, beta=1.0, value=2.0)
    elif condition == "dirichlet":
        spec.bc.dirichlet("T", node_indices=last, value=1.0)
    solution = ds.pdesolve(
        spec.build(),
        ds.JNDF.stock(linear="dense", **{"rtol": 1e-8, "atol": 1e-10}),
        (0.0, 0.5),
        spec.assemble_state({}),
    )
    return np.asarray(solution.ys[-1])


def _cell_and_area_averaged_coefficients(mesh):
    """A coefficient varying per cell of a TRI3 mesh, and its area average at each node."""
    points, cells = np.asarray(mesh.points), np.asarray(mesh.cells)
    a, b, c = (points[cells[:, k]] for k in range(3))
    u, v = b - a, c - a
    area = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    per_cell = 1.5 + (a[:, 0] + b[:, 0] + c[:, 0]) / 3.0 + np.arange(len(cells)) % 2
    weighted, total = np.zeros(len(points)), np.zeros(len(points))
    for k in range(3):
        np.add.at(weighted, cells[:, k], per_cell * area)
        np.add.at(total, cells[:, k], area)
    return per_cell, weighted / total


@pytest.mark.parametrize(
    ("mesh", "reaction", "per_cell"),
    [
        (lambda: interval_mesh(8, 0.0, 1.0), None, False),
        (lambda: rectangle_mesh(3, 3, 1.0, 1.0, ele_type="TRI3"), None, False),
        (lambda: rectangle_mesh(3, 3, 1.0, 1.0, ele_type="TRI3"), 0.5, False),
        (lambda: rectangle_mesh(3, 3, 1.0, 1.0, ele_type="TRI3"), None, True),
    ],
    ids=["line2", "tri3", "tri3-reaction", "tri3-per-cell"],
)
def test_the_residual_form_on_a_mesh_matches_the_right_hand_side_form(mesh, reaction, per_cell):
    """On a mesh, ``2 q.dt(T) - lap(T) - 1 (+ k q.T) = 0`` solves to the same state as
    ``T' = (lap(T) + 1 (- k q.T)) / 2``: the time-derivative coefficient and every residual term
    count, a field read at the quadrature points included. A coefficient given per cell solves
    to the same state as its area average at the nodes."""
    domain = mesh()
    if per_cell:
        cells, nodes = _cell_and_area_averaged_coefficients(domain)
        residual = _heat_end_state(domain, residual_form=True, coeff=cells)
        explicit = _heat_end_state(domain, residual_form=True, coeff=nodes)
    else:
        residual = _heat_end_state(domain, residual_form=True, coeff=2.0, reaction=reaction)
        explicit = _heat_end_state(domain, residual_form=False, coeff=2.0, reaction=reaction)
    # measured 0.0 (line2), 2.9e-15 (tri3) and 2.1e-14 (tri3-reaction), workstation CPU 2026-10-28,
    # and 6.2e-15 (tri3-per-cell), workstation CPU 2026-10-28,
    # pytest tests/public_api/test_recipe_build.py::test_the_residual_form_on_a_mesh_matches_the_right_hand_side_form
    np.testing.assert_allclose(residual, explicit, atol=1e-12)


@pytest.mark.parametrize("condition", ["flux", "robin", "dirichlet"])
def test_the_residual_form_keeps_a_condition_on_the_field_it_writes(condition):
    """On a grid, ``q.dt(T) - lap(T) - 1 = 0`` with a flux, Robin or Dirichlet condition on ``T``
    solves to the same state as ``T' = lap(T) + 1`` with that condition."""
    grid = ds.cartesian_grid_1d(8, 0.0, 1.0)
    residual = _heat_end_state(grid, residual_form=True, coeff=1.0, condition=condition)
    explicit = _heat_end_state(grid, residual_form=False, coeff=1.0, condition=condition)
    # measured 0.0 for all three conditions, workstation CPU 2026-10-28,
    # pytest tests/public_api/test_recipe_build.py::test_the_residual_form_keeps_a_condition_on_the_field_it_writes
    np.testing.assert_allclose(residual, explicit, atol=1e-12)


def _tri6_square():
    """Two six-node triangles on the unit square, the left side tagged."""
    points = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0], [1, 0.5], [0.5, 0.5], [0, 0.5], [0.5, 1]], float
    )
    cells = np.array([[0, 1, 2, 4, 5, 6], [0, 2, 3, 6, 8, 7]])
    return TaggedMesh(
        points, cells, "TRI6", boundary_edges=[[3, 0, 7]], boundary_tags=[1], tag_names={1: "left"}
    )


@pytest.mark.parametrize(
    ("mesh", "side", "loads"),
    [
        (
            lambda: rectangle_mesh(2, 2, 1.0, 1.0, with_boundary_tags=True),
            "left",
            {0: 0.25, 1: 0.5, 2: 0.25},
        ),
        (_tri6_square, "left", {0: 1 / 6, 7: 2 / 3, 3: 1 / 6}),
        (
            lambda: box_mesh(1, 1, 1, 1.0, 1.0, 1.0, with_boundary_tags=True),
            "x_min",
            dict.fromkeys(range(4), 0.25),
        ),
    ],
    ids=["quad4", "tri6", "hex8"],
)
def test_a_flux_on_a_quadrilateral_quadratic_or_hexahedral_mesh_loads_its_consistent_share(
    mesh, side, loads
):
    """A flux density 2 through a unit side of a QUAD4, TRI6 or HEX8 mesh adds ``2 int N_i dS`` to
    each side node's residual row and nothing elsewhere, so the injected total is 2."""
    domain = mesh()
    spec = ds.problem(f"inflow_{domain.ele_type}").domain("plate", domain)
    spec.field("T", mass="differential", scale=1.0)
    spec.equation(writes="T", reads=("T",))(lambda q: q.ops.laplacian(q.T))
    spec.bc.flux("T", tag=side, value=2.0)
    residual = _residual(spec.build(), spec.assemble_state({}))
    expected = np.zeros(domain.num_nodes)
    expected[list(loads)] = 2.0 * np.array(list(loads.values()))
    # measured 0.0 on all three meshes (total 2.0), workstation CPU 2026-10-28,
    # pytest tests/public_api/test_recipe_build.py::test_a_flux_on_a_quadrilateral_quadratic_or_hexahedral_mesh_loads_its_consistent_share
    np.testing.assert_allclose(residual, expected, atol=1e-12)
