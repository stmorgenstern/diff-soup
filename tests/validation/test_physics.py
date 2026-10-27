"""Physics validation: representative models against analytical invariants (periodic heat and
a family exchange graph)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import diff_soup as ds
from diff_soup.domains import cartesian_grid_1d

#: Solver settings (requested tolerances, not asserted bounds).
_TIGHT = {"rtol": 1e-10, "atol": 1e-12}


def _heat(cells, diffusivity=0.5, end=0.4):
    """``(nodes, start, end state)`` of ``T' = D T_xx`` on a periodic ``[0, 2 pi]``."""
    grid = cartesian_grid_1d(cells, 0.0, 2.0 * np.pi, method="fvm")
    spec = ds.problem(f"periodic_heat_{cells}").domain("line", grid)
    spec.field("temperature", mass="differential", scale=1.0, init=lambda x: 0.3 + jnp.sin(x))
    spec.equation(writes="temperature", reads=("temperature",))(
        lambda q: diffusivity * q.ops.laplacian(q.temperature)
    )
    spec.bc.periodic("temperature")
    start = spec.assemble_state({})
    solution = ds.pdesolve(
        spec.build(),
        ds.JNDF.stock(linear="dense", **_TIGHT),
        (0.0, end),
        start,
        saveat=ds.SaveAt(t1=True),
    )
    return np.asarray(grid.nodes), np.asarray(start), np.asarray(solution.ys[-1])


# measured 4.7e-11 from the semi-discrete solution and 3.3e-16 on the mean, CPU VM 2026-10-22,
# pytest tests/validation/test_physics.py::test_periodic_heat_follows_its_exact_semi_discrete_decay_and_conserves_its_mean
_SEMI, _MEAN = 1e-9, 1e-13


def test_periodic_heat_follows_its_exact_semi_discrete_decay_and_conserves_its_mean():
    """The sine mode of a 32-cell periodic finite-volume heat equation decays at the discrete
    rate ``D (4 / h^2) sin^2(h / 2)`` exactly, and the mean temperature stays put."""
    nodes, start, end = _heat(32)
    h = 2.0 * np.pi / 32
    rate = 0.5 * 4.0 / h**2 * np.sin(h / 2.0) ** 2
    np.testing.assert_allclose(end, 0.3 + (start - 0.3) * np.exp(-rate * 0.4), atol=_SEMI)
    np.testing.assert_allclose(end.mean(), start.mean(), atol=_MEAN)
    np.testing.assert_allclose(start, 0.3 + np.sin(nodes), atol=_MEAN)


# measured errors 5.23e-4 (32 cells) and 1.31e-4 (64 cells) against the continuum solution,
# a ratio of 3.98, CPU VM 2026-10-22,
# pytest tests/validation/test_physics.py::test_the_periodic_heat_discretisation_converges_at_second_order
_ORDER = (3.9, 4.1)


def test_the_periodic_heat_discretisation_converges_at_second_order():
    """Halving the cell size divides the error against ``0.3 + e^{-D t} sin x`` by four."""
    errors = []
    for cells in (32, 64):
        nodes, _, end = _heat(cells)
        errors.append(np.max(np.abs(end - (0.3 + np.exp(-0.2) * np.sin(nodes)))))
    assert _ORDER[0] <= errors[0] / errors[1] <= _ORDER[1]


def _path(name, edges=((0, 1), (1, 2)), directed=False, alternative=False):
    """``(problem, view, state)``: three members holding ``c = (1, 4, 2)`` on a weighted graph."""
    spec = ds.problem(name).domain("cell", cartesian_grid_1d(1, 0.0, 1.0, method="fvm"))
    spec.field("macro", mass="differential", scale=1.0)
    spec.equation(writes="macro", reads=("macro",))(lambda q: jnp.zeros_like(q.macro))
    members = spec.family(
        "particles", count=3, domain=cartesian_grid_1d(1, 0.0, 1.0, method="fvm"), scale={"c": 1.0}
    )
    members.field_decl("c", mass="differential")
    members.equation(writes="c", reads=("c",))(lambda q, member: jnp.zeros_like(q.c))
    members.graph(
        "contacts", edges=jnp.asarray(edges), directed=directed, weights=jnp.asarray([2.0, 3.0])
    )

    def exchange(source, target, edge):
        extra = 0.25 * (target.c + source.c) if alternative else 0.0
        return edge.weight * (target.c - source.c) + extra

    members.graph_exchange(
        graph="contacts", writes="c", reads=("c",), weight_reads=("c",), name="exchange"
    )(exchange)
    problem = spec.build()
    view = problem.family_view("particles")
    state = jnp.zeros((problem.state_size(),))
    slab = view.extract(state).at[:, view.field_slice("c")].set(jnp.asarray([[1.0], [4.0], [2.0]]))
    return problem, view, state.at[view.base : view.base + slab.size].set(slab.reshape(-1))


def _exchange_rows(problem, view, state, t=0.0):
    residual = problem.physics_residual(state, jnp.zeros((0,)), jnp.asarray(t))[0]
    return np.asarray(view.field(residual, "c")[:, 0])


# measured 0.0 on the right-hand sides, 5.5e-8 from the matrix exponential at rtol 1e-9 and
# 8.9e-16 on the member total, CPU VM 2026-10-22,
# pytest tests/validation/test_physics.py::test_a_family_graph_exchange_is_its_weighted_laplacian_and_conserves_the_total
_ROWS, _FLOW, _TOTAL = 1e-12, 1e-6, 1e-12
_LAPLACIAN = np.asarray([[2.0, -2.0, 0.0], [-2.0, 5.0, -3.0], [0.0, -3.0, 3.0]])


def test_a_family_graph_exchange_is_its_weighted_laplacian_and_conserves_the_total():
    """On the path 0-1-2 with weights 2 and 3 the exchange rows are ``L c``, the members
    evolve as ``exp(t L) c`` and their total stays 7."""
    problem, view, state = _path("graph_path")
    c0 = np.asarray([1.0, 4.0, 2.0])
    np.testing.assert_allclose(_exchange_rows(problem, view, state), _LAPLACIAN @ c0, atol=_ROWS)
    solver = ds.JNDF.stock(linear="dense", **{"rtol": 1e-9, "atol": 1e-11})
    solution = ds.pdesolve(problem, solver, (0.0, 0.05), state, saveat=ds.SaveAt(t1=True))
    members = np.asarray(view.field(solution.ys[-1], "c")[:, 0])
    values, vectors = np.linalg.eigh(_LAPLACIAN)
    expected = vectors @ (np.exp(0.05 * values) * (vectors.T @ c0))
    np.testing.assert_allclose(members, expected, atol=_FLOW)
    np.testing.assert_allclose(members.sum(), 7.0, atol=_TOTAL)


def test_a_directed_graph_exchange_keeps_its_edge_orientation():
    """A directed graph with an exchange that is not antisymmetric in its endpoints gives one
    set of rows for its edges and another when every edge is reversed, and still conserves
    the total."""
    rows = {
        edges: _exchange_rows(*_path(f"directed_{i}", edges, True, True), t=0.5)
        for i, edges in enumerate((((1, 2), (0, 1)), ((2, 1), (1, 0))))
    }
    np.testing.assert_allclose(rows[((1, 2), (0, 1))], [-4.75, 12.25, -7.5], atol=_ROWS)
    np.testing.assert_allclose(rows[((2, 1), (1, 0))], [-7.75, 13.25, -5.5], atol=_ROWS)
    np.testing.assert_allclose([r.sum() for r in rows.values()], 0.0, atol=_ROWS)
