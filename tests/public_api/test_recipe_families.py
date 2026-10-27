"""Recipe API: family recipes, per-member parameters, member boundary flux and its clock,
surface projection, declared stencil radius, scalar member fields and family graphs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import diff_soup as ds
from diff_soup.domains import rectangle_mesh

# measured 0.0 between the sparse and dense member Jacobians, CPU VM 2026-10-22,
# pytest tests/public_api/test_recipe_families.py::test_a_declared_stencil_radius_widens_the_member_tile_to_the_composed_stencil;
# the residual rows are exact sums of the authored constants
_EXACT = 1e-12
_GRID = ds.cartesian_grid_1d


def _cell(name, cells=2, time_scale=None):
    spec = ds.problem(name).domain("cell", _GRID(cells, 0.0, 1.0, method="fvm"))
    if time_scale is not None:
        spec.time_scale(time_scale)
    spec.field("m", mass="differential", scale=1.0, init=lambda x: jnp.ones_like(x))
    return spec


def _residual(problem, state):
    return problem.physics_residual(state, jnp.zeros((0,)), jnp.asarray(0.0, state.dtype))[0]


def _declared(problem, state):
    """``(sparse, dense, declared)``: the assembled sparse Jacobian, the dense one, the pattern."""
    lam, t = jnp.zeros((0,)), jnp.asarray(0.0)
    values, *_, topology = problem.physics_jacobian_blocks(state, lam, t)
    size = int(problem.state_size())
    sparse, declared = np.zeros((size, size)), np.zeros((size, size), dtype=bool)
    np.add.at(sparse, (topology.rows, topology.cols), np.asarray(values))
    declared[topology.rows, topology.cols] = True
    return sparse, np.asarray(jax.jacfwd(lambda v: _residual(problem, v))(state)), declared


def test_member_parameters_batch_into_the_member_rows_and_the_member_pattern():
    """Two members of rate 1 and 2 on two cells under a three-cell macro field give rows
    ``-1`` (macro and first member) and ``-2``, and the declared pattern holds the diagonal and
    each member's full two-by-two tile."""
    spec = _cell("family_rates", cells=3)
    spec.equation(writes="m", reads=("m",))(lambda q: -q.m)
    members = spec.family(
        "particles",
        count=2,
        domain=_GRID(2, 0.0, 1.0, method="fvm"),
        params={"rate": [1.0, 2.0]},
        scale={"c": 1.0},
        residual_scale={"c": 1.0},
    )
    members.field_decl("c", mass="differential")
    members.equation(writes="c", reads=("c",))(lambda q, member: -member.rate * q.c)
    problem = spec.build()
    state = jnp.ones((problem.state_size(),))
    np.testing.assert_allclose(_residual(problem, state), [-1.0] * 5 + [-2.0] * 2, atol=_EXACT)
    *_, topology = problem.physics_jacobian_blocks(state, jnp.zeros((0,)), jnp.asarray(0.0))
    assert topology.rows.size == 7 + 2 * 2 * 2


@pytest.mark.parametrize("clock", ["argument", "accessor"])
def test_a_member_boundary_flux_reads_si_time_whatever_the_time_scale(clock):
    """A member flux authored as the time itself sees 50 s at internal 0.5 with ``time_ref``
    100 and at internal 50 without, through its argument or ``q.t``."""

    def member_rows(time_scale, t_internal):
        spec = _cell(f"clock_{clock}_{int(time_scale)}", time_scale=time_scale)
        spec.equation(writes="m", reads=("m",))(lambda q: jnp.zeros_like(q.m))
        members = spec.family(
            "particles",
            count=1,
            domain=_GRID(2, 0.0, 1.0, method="fvm"),
            scale={"c": 1.0},
            residual_scale={"c": 1.0},
        )
        members.field_decl("c", mass="differential")
        members.equation(writes="c", reads=("c",))(lambda q, member: jnp.zeros_like(q.c))
        flux = (lambda q, time: time) if clock == "argument" else (lambda q, time: q.t)
        members.bc.flux("c", node_indices=[1], fn=flux)
        problem = spec.build()
        state = jnp.full((problem.state_size(),), 0.5)
        residual, _ = problem.physics_residual(state, jnp.zeros((0,)), jnp.asarray(t_internal))
        return np.asarray(problem.family_view("particles").field(residual, "c"))

    scaled = member_rows(100.0, 0.5)
    np.testing.assert_allclose(scaled, member_rows(1.0, 50.0), atol=_EXACT)
    np.testing.assert_allclose(scaled[:, 1], 100.0, atol=_EXACT)  # 50 s x face area / volume


@pytest.mark.parametrize("projected", ["c", "d"])
def test_a_surface_projection_reads_the_projected_fields_own_surface_node(projected):
    """Projecting the surface of member field ``c`` or ``d`` onto the cell (one alias, one
    mean) reads that field's last node, and the declared pattern covers the dense Jacobian."""
    spec = _cell(f"surface_{projected}", cells=3)
    members = spec.family(
        "particles",
        count=2,
        domain=_GRID(2, 0.0, 1.0, method="fvm"),
        params={"rate": [1.0, 2.0]},
        scale={"c": 1.0, "d": 1.0},
        residual_scale={"c": 1.0, "d": 1.0},
    )
    members.field_decl("c", mass="differential")
    members.field_decl("d", mass="differential")
    members.equation(writes="c", reads=("c",))(lambda q, member: -member.rate * q.c)
    members.equation(writes="d", reads=("d",))(lambda q, member: -q.d)
    members.couple(members.field(projected).surface(), onto="cell", at=[0, 1], as_="surf")
    members.couple(members.field(projected).surface(), onto="cell", at=[1, 1], reduce="mean")
    spec.scales(state={f"particles_{projected}_at_cell": 1.0})
    spec.equation(writes="m", reads=("m", "surf"))(lambda q: -q.m + q.surf)
    problem = spec.build()
    state = jnp.arange(float(problem.state_size()))
    residual = np.asarray(_residual(problem, state))
    surface = np.asarray(problem.family_view("particles").field(state, projected))[:, -1]
    np.testing.assert_allclose(residual[:3], -np.arange(3.0) + [*surface, 0.0], atol=_EXACT)
    np.testing.assert_allclose(
        residual[3:6], -np.arange(3.0, 6.0) + [0.0, surface.mean(), 0.0], atol=_EXACT
    )
    _, dense, declared = _declared(problem, state)
    assert not np.any((np.abs(dense) > 0) & ~declared)


def _composed_laplacian(domain, *, composed=True, **family):
    spec = _cell(f"wide_stencil_{composed}_{family.get('stencil_radius')}")
    spec.equation(writes="m", reads=("m",))(lambda q: -q.m)
    members = spec.family(
        "particles",
        count=2,
        domain=domain,
        params={"rate": [1.0, 2.0]},
        scale={"c": 1.0},
        residual_scale={"c": 1.0},
        **family,
    )
    members.field_decl("c", mass="differential")

    def member(q, member):
        if not composed:
            return -member.rate * q.c
        return -member.rate * q.ops.laplacian(q.ops.laplacian(q.c)) - q.c

    members.equation(writes="c", reads=("c",))(member)
    return spec


def test_a_declared_stencil_radius_widens_the_member_tile_to_the_composed_stencil():
    """With ``stencil_radius=2`` the declared pattern of a composed laplacian covers the dense
    Jacobian and adds the distance-two pairs (2 members x 2 x 5); the radius is validated
    where it is authored and needs a structured member grid."""
    grid = _GRID(7, 0.0, 1.0, method="fvm")
    default = _composed_laplacian(grid).build()
    wide = _composed_laplacian(grid, stencil_radius=2).build()
    state = jnp.linspace(0.3, 1.7, wide.state_size())
    sparse, dense, declared = _declared(wide, state)
    assert not np.any((np.abs(dense) > 0) & ~declared)
    np.testing.assert_allclose(sparse, dense, rtol=0.0, atol=_EXACT)
    assert int(declared.sum()) == int(_declared(default, state)[2].sum()) + 20
    with pytest.raises(ValueError, match="stencil_radius must be at least 1"):
        _composed_laplacian(grid, stencil_radius=0)
    with pytest.raises(TypeError, match="stencil_radius must be None or a positive int"):
        _composed_laplacian(grid, stencil_radius=True)
    with pytest.raises(ValueError, match="not a structured grid"):
        _composed_laplacian(
            rectangle_mesh(2, 2, 1.0, 1.0), composed=False, stencil_radius=2
        ).build()


def test_a_scalar_algebraic_member_field_solves_and_projects_to_its_closed_form():
    """A per-member scalar ``j = k c_surface`` (refused unless algebraic and one-dimensional)
    projects onto the cell; uniform members give ``c = c0 e^{(k-1)t}``, ``j = k c`` and
    ``m = c0 e^{-t} (e^{kt} - 1)``."""
    rates, c0, end = np.array([0.5, 2.0]), np.array([1.0, 2.0]), 0.5
    spec = _cell("scalar_member")
    members = spec.family(
        "particles",
        count=2,
        domain=_GRID(3, 0.0, 1.0, method="fvm"),
        params={"k": rates},
        scale={"c": 1.0, "j": 1.0},
        residual_scale={"c": 1.0, "j": 1.0},
    )
    members.field_decl("c", mass="differential")
    with pytest.raises(ValueError, match="per-member scalar.*must be mass='algebraic'"):
        members.field_decl("bad", mass="differential", shape=(1,))
    with pytest.raises(ValueError, match="multi-dim member field"):
        members.field_decl("bad", mass="algebraic", shape=(2,))
    members.field_decl("j", mass="algebraic", shape=(1,))
    members.equation(writes="c", reads=("c", "j"))(lambda q, member: -q.c + q.j)
    members.equation(writes="j", reads=("c", "j"))(lambda q, member: q.j - member.k * q.c[-1])
    members.couple(members.field("j").surface(), onto="cell", at=[0, 1], as_="rate")
    spec.equation(writes="m", reads=("m", "rate"))(lambda q: -q.m + q.rate)
    problem = spec.build()
    initial = problem.assemble_state(
        {
            "m": jnp.zeros(2),
            "particles.c": jnp.repeat(jnp.asarray(c0)[:, None], 3, axis=1),
            "particles.j": jnp.asarray(rates * c0)[:, None],
        }
    )
    final = ds.pdesolve(
        problem, ds.JNDF.stock(linear="dense", **{"rtol": 1e-9, "atol": 1e-11}), (0.0, end), initial
    ).ys[-1]
    view, c_end = problem.family_view("particles"), c0 * np.exp((rates - 1.0) * end)
    got = np.concatenate(
        [
            np.asarray(view.field(final, "c"))[:, 0],
            np.asarray(view.field(final, "j"))[:, 0],
            np.asarray(final[:2]),
        ]
    )
    expected = np.concatenate(
        [c_end, rates * c_end, c0 * np.exp(-end) * (np.exp(rates * end) - 1.0)]
    )
    # measured 3.6e-9 relative at rtol 1e-9, CPU VM 2026-10-22,
    # pytest tests/public_api/test_recipe_families.py::test_a_scalar_algebraic_member_field_solves_and_projects_to_its_closed_form
    np.testing.assert_allclose(got, expected, rtol=1e-7)


def _graph_recipe(name="family_graph", **graph):
    """Three members on a path exchanging ``c`` across a weighted graph (not built)."""
    spec = _cell(name, cells=1)
    spec.equation(writes="m", reads=("m",))(lambda q: jnp.zeros_like(q.m))
    members = spec.family(
        "particles",
        count=3,
        domain=_GRID(1, 0.0, 1.0, method="fvm"),
        params={"rate": jnp.asarray([1.0, 1.5, 2.0])},
        scale={"c": 1.0, "temperature": 1.0},
    )
    members.field_decl("c", mass="differential")
    members.field_decl("temperature", mass="differential")
    members.equation(writes="c", reads=("c",))(lambda q, member: jnp.zeros_like(q.c))
    members.equation(writes="temperature", reads=("temperature",))(
        lambda q, member: jnp.zeros_like(q.temperature)
    )
    weight_reads = graph.pop("weight_reads", ("c",))
    result = members.graph(
        "contacts",
        edges=jnp.asarray(graph.pop("edges", ((0, 1), (1, 2)))),
        directed=False,
        weights=graph.pop("weights", jnp.asarray([2.0, 3.0])),
    )
    members.graph_exchange(
        graph="contacts", writes="c", reads=("c",), weight_reads=weight_reads, name="exchange"
    )(lambda source, target, edge: edge.weight * (target.c - source.c))
    return spec, result


def test_a_family_graph_canonicalizes_its_edges_and_validates_its_declarations():
    """Reversed undirected edges are stored sorted with their weights; malformed edges and
    weights, an undeclared state weight read, and malformed exchanges and families are
    refused naming the fault."""
    spec, graph = _graph_recipe(edges=((1, 0), (2, 1)), weights=(3.0, 2.0))
    np.testing.assert_array_equal(graph.edges, [[0, 1], [1, 2]])
    np.testing.assert_allclose(graph.weights, [2.0, 3.0], atol=_EXACT)
    assert (graph.member_count, graph.directed) == (3, False)
    assert spec.build().family_view("particles").graph("contacts").name == "contacts"
    for keywords, match in (
        ({"edges": jnp.asarray([0, 1])}, "shape"),
        ({"edges": jnp.asarray([[0.0], [1.0]])}, "integer"),
        ({"edges": jnp.asarray([[0], [3]])}, "range"),
        ({"edges": jnp.asarray([[1], [1]])}, "self-loop"),
        ({"edges": jnp.asarray([[0, 1], [1, 0]])}, "duplicate"),
        ({"weights": (1.0, 2.0, 3.0)}, "weight array length"),
        ({"weights": jnp.asarray([2, 3])}, "floating-point dtype"),
    ):
        pytest.raises((TypeError, ValueError), _graph_recipe, **keywords).match(match)
    with pytest.raises(ValueError, match="temperature.*weight_reads"):
        weight = lambda s, t, e: s.p.rate * (1.0 + s.temperature + t.temperature)  # noqa: E731
        _graph_recipe("graph_reads", weights=weight, weight_reads=("c",))[0].build()
    template = spec.to_equation_set().families[0]
    base = dict(
        name="x", graph="contacts", writes="c", reads=("c",), weight_reads=(), fn=lambda *x: 0
    )
    exchange = ds.FamilyGraphExchange(**base)

    def family(**changes):
        return ds.Family(
            **(
                dict(
                    name="direct",
                    count=3,
                    domain=template.domain,
                    member_variables=template.member_variables,
                    graphs=(graph,),
                    graph_exchanges=(exchange,),
                )
                | changes
            )
        )

    for constructor, changes, match in (
        (ds.FamilyGraphExchange, base | {"reads": ("c", "c")}, "duplicate"),
        (ds.FamilyGraphExchange, base | {"fn": 1.0}, "callable"),
        (family, {"count": 2}, "member_count 3"),
        (family, {"graphs": (graph, graph)}, "duplicate graph"),
        (
            family,
            {"graph_exchanges": (ds.FamilyGraphExchange(**(base | {"graph": "missing"})),)},
            "unknown graph",
        ),
        (
            family,
            {"graph_exchanges": (ds.FamilyGraphExchange(**(base | {"writes": "missing"})),)},
            "unknown member field",
        ),
    ):
        pytest.raises((TypeError, ValueError), constructor, **changes).match(match)
